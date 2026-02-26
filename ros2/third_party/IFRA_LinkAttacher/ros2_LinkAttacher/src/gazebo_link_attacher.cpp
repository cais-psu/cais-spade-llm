/*
# ===================================== COPYRIGHT ===================================== #
#                                                                                       #
#  IFRA (Intelligent Flexible Robotics and Assembly) Group, CRANFIELD UNIVERSITY        #
#  Created on behalf of the IFRA Group at Cranfield University, United Kingdom          #
#  E-mail: IFRA@cranfield.ac.uk                                                         #
#                                                                                       #
#  Licensed under the Apache-2.0 License.                                               #
#  You may not use this file except in compliance with the License.                     #
#  You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0  #
#                                                                                       #
#  Unless required by applicable law or agreed to in writing, software distributed      #
#  under the License is distributed on an "as-is" basis, without warranties or          #
#  conditions of any kind, either express or implied. See the License for the specific  #
#  language governing permissions and limitations under the License.                    #
#                                                                                       #
#  IFRA Group - Cranfield University                                                    #
#  AUTHORS: Mikel Bueno Viso - Mikel.Bueno-Viso@cranfield.ac.uk                         #
#           Dr. Seemal Asif  - s.asif@cranfield.ac.uk                                   #
#           Prof. Phil Webb  - p.f.webb@cranfield.ac.uk                                 #
#                                                                                       #
#  Date: May, 2023.                                                                     #
#                                                                                       #
# ===================================== COPYRIGHT ===================================== #

# ======= CITE OUR WORK ======= #
# You can cite our work with the following statement:
# IFRA-Cranfield (2023) IFRA Gazebo-ROS2 Link Attacher. URL: https://github.com/IFRA-Cranfield/IFRA_LinkAttacher.
*/

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Entity.hh>
#include <gazebo/physics/Light.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/PhysicsEngine.hh>

#include <gazebo_ros/node.hpp>
#include <memory>
#include <algorithm>

#include "gazebo_ros/conversions/builtin_interfaces.hpp"
#include "gazebo_ros/conversions/geometry_msgs.hpp"

#include "ros2_linkattacher/gazebo_link_attacher.hpp"   // INCLUDE HADER FILE.
#include <linkattacher_msgs/srv/attach_link.hpp>        // INCLUDE ROS2 SERVICE.
#include <linkattacher_msgs/srv/detach_link.hpp>        // INCLUDE ROS2 SERVICE.

// GLOBAL VARIABLE:
std::vector<JointSTRUCT> GV_joints;
static unsigned long GV_joint_counter = 0;

namespace gazebo_ros
{

class GazeboLinkAttacherPrivate
{
public:

  // ATTACH (ROS2 service):
  void Attach(
    linkattacher_msgs::srv::AttachLink::Request::SharedPtr _req,
    linkattacher_msgs::srv::AttachLink::Response::SharedPtr _res);

  // DETACH (ROS2 service):
  void Detach(
    linkattacher_msgs::srv::DetachLink::Request::SharedPtr _req,
    linkattacher_msgs::srv::DetachLink::Response::SharedPtr _res);

  // World pointer from Gazebo.
  gazebo::physics::WorldPtr world_;

  /// ROS node for communication, managed by gazebo_ros.
  gazebo_ros::Node::SharedPtr ros_node_;

  // ROS services to handle requests for attach/detach.
  rclcpp::Service<linkattacher_msgs::srv::AttachLink>::SharedPtr attach_link_service_;
  rclcpp::Service<linkattacher_msgs::srv::DetachLink>::SharedPtr detach_link_service_;

  // getJoint function:
  bool getJoint(std::string M1, std::string L1, std::string M2, std::string L2, JointSTRUCT &joint);

};

GazeboLinkAttacher::GazeboLinkAttacher()
: impl_(std::make_unique<GazeboLinkAttacherPrivate>())
{
}

GazeboLinkAttacher::~GazeboLinkAttacher()
{
}

void GazeboLinkAttacher::Load(gazebo::physics::WorldPtr _world, sdf::ElementPtr _sdf)
{
  
  // Gazebo WORLD:
  impl_->world_ = _world;

  // ROS2 NODE:
  impl_->ros_node_ = gazebo_ros::Node::Get(_sdf);

  // ROS2 SERVICE SERVERS:
  impl_->attach_link_service_ =
    impl_->ros_node_->create_service<linkattacher_msgs::srv::AttachLink>(
    "ATTACHLINK", std::bind(
      &GazeboLinkAttacherPrivate::Attach, impl_.get(),
      std::placeholders::_1, std::placeholders::_2));
  impl_->detach_link_service_ =
    impl_->ros_node_->create_service<linkattacher_msgs::srv::DetachLink>(
    "DETACHLINK", std::bind(
      &GazeboLinkAttacherPrivate::Detach, impl_.get(),
      std::placeholders::_1, std::placeholders::_2));

}

void GazeboLinkAttacherPrivate::Attach(
  linkattacher_msgs::srv::AttachLink::Request::SharedPtr _req,
  linkattacher_msgs::srv::AttachLink::Response::SharedPtr _res)
{

  // If this exact pair already exists, re-attach and return success.
  JointSTRUCT j;
  if (this->getJoint(_req->model1_name, _req->link1_name, _req->model2_name, _req->link2_name, j)){
    if (j.joint) {
      j.joint->Attach(j.l1, j.l2);
    }
    _res->success = true;
    _res->message = "ATTACHED (existing): {MODEL , LINK} -> {" + _req->model1_name + " , " + _req->link1_name + "} -- {" + _req->model2_name + " , " + _req->link2_name + "}.";
    return;
  }

  // Get the first link:
  gazebo::physics::ModelPtr model1 = world_->ModelByName(_req->model1_name);
  if (!model1) {
    _res->success = false;
    _res->message = "Failed to find model with name: " + _req->model1_name;
    return;
  }
  gazebo::physics::LinkPtr link1 = model1->GetLink(_req->link1_name);
  if (!link1) {
    _res->success = false;
    _res->message = "Failed to find link with name: " + _req->link1_name;
    return;
  }

  // Get the second link:
  gazebo::physics::ModelPtr model2 = world_->ModelByName(_req->model2_name);
  if (!model2) {
    _res->success = false;
    _res->message = "Failed to find model with name: " + _req->model2_name;
    return;
  }
  gazebo::physics::LinkPtr link2 = model2->GetLink(_req->link2_name);
  if (!link2) {
    _res->success = false;
    _res->message = "Failed to find link with name: " + _req->link2_name;
    return;
  }

  // Prevent a single link from being attached to multiple links at once.
  for (const auto &existing : GV_joints) {
    bool first_busy =
      (existing.model1 == _req->model1_name && existing.link1 == _req->link1_name) ||
      (existing.model2 == _req->model1_name && existing.link2 == _req->link1_name);
    bool second_busy =
      (existing.model1 == _req->model2_name && existing.link1 == _req->link2_name) ||
      (existing.model2 == _req->model2_name && existing.link2 == _req->link2_name);
    if (first_busy || second_busy) {
      _res->success = false;
      _res->message = "One or both links are already attached to another link.";
      return;
    }
  }

  // Create a fixed joint between the two links:
  std::string joint_name = _req->model1_name + "_" + _req->link1_name + "_" + _req->model2_name + "_" + _req->link2_name +
                           "_joint_" + std::to_string(GV_joint_counter++);
  gazebo::physics::JointPtr joint = model1->CreateJoint(joint_name, "revolute", link1, link2);
  // Preserve the current grasp geometry instead of snapping link2 to link1 origin.
  const ignition::math::Pose3d link1_world_pose = link1->WorldPose();
  const ignition::math::Pose3d link2_world_pose = link2->WorldPose();
  const ignition::math::Pose3d relative_pose = link1_world_pose.Inverse() * link2_world_pose;
  joint->Attach(link1, link2);
  joint->Load(link1, link2, relative_pose);
  joint->SetProvideFeedback(true);

  joint->SetAxis(0, ignition::math::Vector3d(1, 0, 0));
  joint->SetUpperLimit(0, 0);
  joint->SetLowerLimit(0, 0);
  joint->SetEffortLimit(0, 0);
  joint->SetDamping(1, 1.0);

  joint->Init();
  model1->Update();

  JointSTRUCT joint_entry;
  joint_entry.model1 = _req->model1_name;
  joint_entry.model2 = _req->model2_name;
  joint_entry.link1 = _req->link1_name;
  joint_entry.link2 = _req->link2_name;
  joint_entry.m1 = model1;
  joint_entry.m2 = model2;
  joint_entry.l1 = link1;
  joint_entry.l2 = link2;
  joint_entry.joint = joint;

  GV_joints.push_back(joint_entry);

  // Set the success and message in the response:
  _res->success = true;
  _res->message = "ATTACHED: {MODEL , LINK} -> {" + _req->model1_name + " , " + _req->link1_name + "} -- {" + _req->model2_name + " , " + _req->link2_name + "}.";

}

void GazeboLinkAttacherPrivate::Detach(
  linkattacher_msgs::srv::DetachLink::Request::SharedPtr _req,
  linkattacher_msgs::srv::DetachLink::Response::SharedPtr _res)
{

  // CHECK if -> Joint already exists in GV_joints:
  JointSTRUCT j;
  if (this->getJoint(_req->model1_name, _req->link1_name, _req->model2_name, _req->link2_name, j)){
    if (j.joint) {
      j.joint->Detach();
    }
    _res->success = true;
    _res->message = "DETACHED: {MODEL , LINK} -> {" + _req->model1_name + " , " + _req->link1_name + "} -- {" + _req->model2_name + " , " + _req->link2_name + "}.";
    
    // Remove the created joint by name so the same pair can be re-attached later.
    gazebo::physics::ModelPtr model1 = world_->ModelByName(_req->model1_name);
    if (model1 && j.joint) {
      model1->RemoveJoint(j.joint->GetName());
    }

    GV_joints.erase(
      std::remove_if(
        GV_joints.begin(),
        GV_joints.end(),
        [&](const JointSTRUCT &entry) {
          return entry.model1 == _req->model1_name &&
                 entry.link1 == _req->link1_name &&
                 entry.model2 == _req->model2_name &&
                 entry.link2 == _req->link2_name;
        }),
      GV_joints.end());
    
    return;
  } else {
    _res->success = false;
    _res->message = "DETACHED -- ERROR (Joint does not exist!): {MODEL , LINK} -> {" + _req->model1_name + " , " + _req->link1_name + "} -- {" + _req->model2_name + " , " + _req->link2_name + "}.";
  }

}

bool GazeboLinkAttacherPrivate::getJoint(std::string M1, std::string L1, std::string M2, std::string L2, JointSTRUCT &joint)
  {
    JointSTRUCT j;
    for(std::vector<JointSTRUCT>::iterator it = GV_joints.begin(); it != GV_joints.end(); ++it){
      j = *it;
      if ((j.model1.compare(M1) == 0) && (j.model2.compare(M2) == 0) && (j.link1.compare(L1) == 0) && (j.link2.compare(L2) == 0)){
        joint = j;
        return true;
      }
    }
    return false;
  }

GZ_REGISTER_WORLD_PLUGIN(GazeboLinkAttacher)

}  // namespace gazebo_ros

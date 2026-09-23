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
#include <gazebo/common/Events.hh>
#include <gazebo/physics/Entity.hh>
#include <gazebo/physics/Light.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/World.hh>
#include <gazebo/physics/PhysicsEngine.hh>

#include <gazebo_ros/node.hpp>
#include <memory>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <future>
#include <mutex>

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

  void QueueAttach(
    linkattacher_msgs::srv::AttachLink::Request::SharedPtr request,
    linkattacher_msgs::srv::AttachLink::Response::SharedPtr response);
  void QueueDetach(
    linkattacher_msgs::srv::DetachLink::Request::SharedPtr request,
    linkattacher_msgs::srv::DetachLink::Response::SharedPtr response);
  bool RunOnPhysicsThread(std::function<void()> operation);
  void OnUpdate();

  struct PendingCommand {
    // 0: pending, 1: running, 2: cancelled. A timed-out pending operation never runs.
    std::atomic<int> state{0};
    std::function<void()> operation;
    std::promise<void> completion;
  };
  std::mutex command_mutex_;
  std::deque<std::shared_ptr<PendingCommand>> commands_;
  gazebo::event::ConnectionPtr update_connection_;

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

  impl_->update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
    std::bind(&GazeboLinkAttacherPrivate::OnUpdate, impl_.get()));

  // ROS2 SERVICE SERVERS:
  impl_->attach_link_service_ =
    impl_->ros_node_->create_service<linkattacher_msgs::srv::AttachLink>(
    "ATTACHLINK", std::bind(
      &GazeboLinkAttacherPrivate::QueueAttach, impl_.get(),
      std::placeholders::_1, std::placeholders::_2));
  impl_->detach_link_service_ =
    impl_->ros_node_->create_service<linkattacher_msgs::srv::DetachLink>(
    "DETACHLINK", std::bind(
      &GazeboLinkAttacherPrivate::QueueDetach, impl_.get(),
      std::placeholders::_1, std::placeholders::_2));

}

bool GazeboLinkAttacherPrivate::RunOnPhysicsThread(std::function<void()> operation)
{
  auto command = std::make_shared<PendingCommand>();
  command->operation = std::move(operation);
  auto completed = command->completion.get_future();
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    commands_.push_back(command);
  }
  if (completed.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
    int pending = 0;
    if (command->state.compare_exchange_strong(pending, 2)) {
      return false;
    }
    // Already executing: retain the response until the physics mutation completes.
    completed.wait();
  }
  completed.get();
  return true;
}

void GazeboLinkAttacherPrivate::OnUpdate()
{
  std::deque<std::shared_ptr<PendingCommand>> commands;
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    commands.swap(commands_);
  }
  for (const auto &command : commands) {
    int pending = 0;
    if (!command->state.compare_exchange_strong(pending, 1)) {
      continue;
    }
    try {
      command->operation();
      command->completion.set_value();
    } catch (...) {
      command->completion.set_exception(std::current_exception());
    }
  }
}

void GazeboLinkAttacherPrivate::QueueAttach(
  linkattacher_msgs::srv::AttachLink::Request::SharedPtr request,
  linkattacher_msgs::srv::AttachLink::Response::SharedPtr response)
{
  try {
    if (!RunOnPhysicsThread([this, request, response]() { Attach(request, response); })) {
      response->success = false;
      response->message = "Attachment cancelled: Gazebo physics did not acknowledge the request.";
    }
  } catch (const std::exception &error) {
    response->success = false;
    response->message = error.what();
  }
}

void GazeboLinkAttacherPrivate::QueueDetach(
  linkattacher_msgs::srv::DetachLink::Request::SharedPtr request,
  linkattacher_msgs::srv::DetachLink::Response::SharedPtr response)
{
  try {
    if (!RunOnPhysicsThread([this, request, response]() { Detach(request, response); })) {
      response->success = false;
      response->message = "Detachment cancelled: Gazebo physics did not acknowledge the request.";
    }
  } catch (const std::exception &error) {
    response->success = false;
    response->message = error.what();
  }
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

  // The assembly carrier is an attachment hub: its one invisible link retains
  // both fixtures and every assembled component. Other links remain exclusive,
  // so a gripper or payload cannot hold multiple attachments at once.
  const bool first_allows_multiple =
    _req->model1_name == "assembly_board_v1" && _req->link1_name == "link";
  for (const auto &existing : GV_joints) {
    bool first_busy =
      (existing.model1 == _req->model1_name && existing.link1 == _req->link1_name) ||
      (existing.model2 == _req->model1_name && existing.link2 == _req->link1_name);
    bool second_busy =
      (existing.model1 == _req->model2_name && existing.link1 == _req->link2_name) ||
      (existing.model2 == _req->model2_name && existing.link2 == _req->link2_name);
    if (second_busy || (first_busy && !first_allows_multiple)) {
      _res->success = false;
      _res->message = "One or both links are already attached to another link.";
      return;
    }
  }

  // The nominal workflow finishes on this static carrier. Freeze an assembled
  // component and remove its contact response so conservative demonstration
  // collision envelopes cannot push the visible mesh off its configured slot.
  if (first_allows_multiple) {
    link2->SetCollideMode("none");
    link2->SetGravityMode(false);
  }

  // A native fixed joint records the links' current relative position and
  // rotation when Model::CreateJoint attaches them. This prevents a payload
  // from being pulled toward a revolute anchor when the constraint starts.
  std::string joint_name = _req->model1_name + "_" + _req->link1_name + "_" + _req->model2_name + "_" + _req->link2_name +
                           "_joint_" + std::to_string(GV_joint_counter++);
  gazebo::physics::JointPtr joint = model1->CreateJoint(joint_name, "fixed", link1, link2);
  joint->SetProvideFeedback(true);
  joint->Init();

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

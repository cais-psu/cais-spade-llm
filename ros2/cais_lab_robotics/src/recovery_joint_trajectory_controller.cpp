// Read-only owner observations for the existing joint trajectory controller.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

#include <joint_trajectory_controller/joint_trajectory_controller.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace cais_lab_robotics
{
class RecoveryJointTrajectoryController
  : public joint_trajectory_controller::JointTrajectoryController
{
public:
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override
  {
    auto result = JointTrajectoryController::on_configure(previous_state);
    if (result != controller_interface::CallbackReturn::SUCCESS) {
      return result;
    }
    {
      std::lock_guard<std::mutex> guard(observation_mutex_);
      observation_.clear();
      sequence_ = 0;
      command_revision_ = 0;
      last_command_.reset();
      instance_id_ = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    }
    observation_service_ = get_node()->create_service<std_srvs::srv::Trigger>(
      "~/recovery_state",
      [this](std_srvs::srv::Trigger::Request::SharedPtr,
      std_srvs::srv::Trigger::Response::SharedPtr response) {
        if (get_node()->get_current_state().id() !=
          lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
        {
          response->success = false;
          response->message = "Controller is not active";
          return;
        }
        std::lock_guard<std::mutex> guard(observation_mutex_);
        response->success = !observation_.empty();
        response->message = response->success ? observation_ : "Controller update not observed";
      });
    return result;
  }

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override
  {
    auto result = JointTrajectoryController::update(time, period);
    // A reader can delay publication, never the controller update. Consumers
    // reject a stale sequence or timestamp instead of inferring an idle state.
    std::unique_lock<std::mutex> guard(observation_mutex_, std::try_to_lock);
    if (!guard.owns_lock()) {
      return result;
    }
    const auto goal = *rt_active_goal_.readFromRT();
    const auto command = *traj_msg_external_point_ptr_.readFromRT();
    if (command != last_command_) {
      last_command_ = command;
      ++command_revision_;
    }
    const bool active = goal && goal->gh_ && goal->gh_->is_active();
    std::ostringstream row;
    row << std::setprecision(17)
        << "{\"version\":1,\"instance_id\":" << std::quoted(instance_id_)
        << ",\"controller\":" << std::quoted(get_node()->get_node_base_interface()->get_fully_qualified_name())
        << ",\"sequence\":" << ++sequence_
        << ",\"command_revision\":" << command_revision_
        << ",\"simulation_time\":" << time.seconds()
        << ",\"has_active_goal\":" << (active ? "true" : "false")
        << ",\"has_pending_goal\":" << (rt_has_pending_goal_.load() ? "true" : "false")
        << ",\"holding\":" << (rt_is_holding_.load() ? "true" : "false")
        << ",\"joint_names\":[";
    for (size_t i = 0; i < params_.joints.size(); ++i) {
      if (i) row << ',';
      row << std::quoted(params_.joints[i]);
    }
    row << "],\"positions\":[";
    for (size_t i = 0; i < state_current_.positions.size(); ++i) {
      if (!std::isfinite(state_current_.positions[i])) {
        observation_.clear();
        return result;
      }
      if (i) row << ',';
      row << state_current_.positions[i];
    }
    row << "]}";
    observation_ = row.str();
    return result;
  }

private:
  std::mutex observation_mutex_;
  std::string observation_;
  std::string instance_id_;
  uint64_t sequence_{0};
  uint64_t command_revision_{0};
  std::shared_ptr<trajectory_msgs::msg::JointTrajectory> last_command_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr observation_service_;
};
}  // namespace cais_lab_robotics

PLUGINLIB_EXPORT_CLASS(
  cais_lab_robotics::RecoveryJointTrajectoryController,
  controller_interface::ControllerInterface)

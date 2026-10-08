// Native command ownership and observations; not a Gazebo containment certificate.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <joint_trajectory_controller/joint_trajectory_controller.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp_action/create_server.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "cais_lab_robotics/srv/set_recovery_motion_contract.hpp"

namespace cais_lab_robotics
{
class RecoveryJointTrajectoryController
  : public joint_trajectory_controller::JointTrajectoryController
{
  using Contract = srv::SetRecoveryMotionContract;
  using GoalHandle = rclcpp_action::ServerGoalHandle<FollowJTrajAction>;
  static constexpr const char * physical_reason_ =
    "gazebo_physics_containment_and_stopping_unverified";

public:
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override
  {
    const auto result = JointTrajectoryController::on_configure(previous_state);
    if (result != controller_interface::CallbackReturn::SUCCESS) {return result;}
    {
      std::lock_guard<std::mutex> lock(owner_mutex_);
      observation_.clear();
      sequence_ = command_revision_ = contract_revision_ = rejected_commands_ = 0;
      stationary_samples_ = 0;
      last_command_.reset();
      instance_id_ = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
      contract_state_ = "unlocked";
      reservation_token_.clear();
      binding_fingerprint_.clear();
      goal_claimed_ = stop_requested_ = stationary_ = false;
      maximum_period_ = 0.;
      maximum_error_.assign(params_.joints.size(), 0.);
      observed_positions_.clear();
      observed_velocities_.clear();
    }
    // The base callbacks are nonvirtual. Replace both public endpoints so a
    // second ROS sender cannot bypass an armed reservation or stationary hold.
    joint_command_subscriber_.reset();
    joint_command_subscriber_ = get_node()->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "~/joint_trajectory", rclcpp::SystemDefaultsQoS(),
      [this](trajectory_msgs::msg::JointTrajectory::SharedPtr command) {
        std::lock_guard<std::mutex> lock(owner_mutex_);
        if (contract_state_ != "unlocked") {++rejected_commands_; return;}
        topic_callback(command);
      });
    action_server_.reset();
    action_server_ = rclcpp_action::create_server<FollowJTrajAction>(
      get_node()->get_node_base_interface(), get_node()->get_node_clock_interface(),
      get_node()->get_node_logging_interface(), get_node()->get_node_waitables_interface(),
      std::string(get_node()->get_name()) + "/follow_joint_trajectory",
      [this](const rclcpp_action::GoalUUID & uuid,
      std::shared_ptr<const FollowJTrajAction::Goal> goal) {
        std::lock_guard<std::mutex> lock(owner_mutex_);
        if (contract_state_ != "unlocked" &&
          (contract_state_ != "armed" || stationary_ || goal_claimed_ ||
          !observation_fresh() ||
          uuid != controller_goal_id_ || goal->trajectory != trajectory_ ||
          stationary_samples_ < 2 || observed_positions_ != trajectory_.points.front().positions ||
          !goal->path_tolerance.empty() || !goal->goal_tolerance.empty() ||
          goal->goal_time_tolerance.sec != 0 || goal->goal_time_tolerance.nanosec != 0))
        {
          ++rejected_commands_;
          return rclcpp_action::GoalResponse::REJECT;
        }
        const auto response = goal_received_callback(uuid, goal);
        if (response != rclcpp_action::GoalResponse::REJECT && contract_state_ == "armed") {
          goal_claimed_ = true;
          ++contract_revision_;
        }
        return response;
      },
      [this](const std::shared_ptr<GoalHandle> goal) {
        std::lock_guard<std::mutex> lock(owner_mutex_);
        if (contract_state_ != "unlocked" && goal->get_goal_id() != controller_goal_id_) {
          ++rejected_commands_;
          return rclcpp_action::CancelResponse::REJECT;
        }
        const auto response = goal_cancelled_callback(goal);
        if (contract_state_ != "unlocked" && response == rclcpp_action::CancelResponse::ACCEPT) {
          request_stop("owner_goal_cancelled");
        }
        return response;
      },
      [this](const std::shared_ptr<GoalHandle> goal) {
        std::lock_guard<std::mutex> lock(owner_mutex_);
        if (contract_state_ != "unlocked" &&
          (contract_state_ != "armed" || !goal_claimed_ ||
          !observation_fresh() ||
          goal->get_goal_id() != controller_goal_id_ || goal->get_goal()->trajectory != trajectory_ ||
          stationary_samples_ < 2 || observed_positions_ != trajectory_.points.front().positions))
        {
          auto response = std::make_shared<FollowJTrajAction::Result>();
          response->error_code = FollowJTrajAction::Result::INVALID_GOAL;
          response->error_string = "Native recovery reservation changed before acceptance";
          goal->abort(response);
          ++rejected_commands_;
          return;
        }
        goal_accepted_callback(goal);
        if (contract_state_ == "armed") {
          contract_state_ = "active";
          owned_goal_ = goal;
          ++contract_revision_;
        }
      });
    observation_service_ = get_node()->create_service<std_srvs::srv::Trigger>(
      "~/recovery_state", [this](std_srvs::srv::Trigger::Request::SharedPtr,
      std_srvs::srv::Trigger::Response::SharedPtr response) {
        std::lock_guard<std::mutex> lock(owner_mutex_);
        // update(time) comes directly from Gazebo; this node's asynchronous
        // /clock callback may still lag it. Raw telemetry stays available while
        // fresh; command admission retains the stricter simulation-clock check.
        response->success = active() && observation_fresh(false);
        response->message = response->success ? observation_ : "Active controller update unavailable";
      });
    contract_service_ = get_node()->create_service<Contract>(
      "~/recovery_motion_contract", [this](Contract::Request::SharedPtr request,
      Contract::Response::SharedPtr response) {
        std::lock_guard<std::mutex> lock(owner_mutex_);
        response->reason = apply_contract(*request);
        response->accepted = response->reason.empty();
        fill_response(*response);
        if (response->accepted && request->operation == "prepare") {
          response->binding_fingerprint = request->binding_fingerprint;
          response->stationary = request->stationary;
        }
      });
    return result;
  }

  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override
  {
    std::lock_guard<std::mutex> lock(owner_mutex_);
    if (contract_state_ != "unlocked") {request_stop("controller_deactivated");}
    // Reactivation can reuse the base controller's unchanged hold-message
    // pointer. A new owner incarnation must invalidate pre-deactivation proofs
    // independently of joint values or command revisions, retaining any claim.
    instance_id_ = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
    observation_.clear();
    stationary_samples_ = 0;
    observed_positions_.clear();
    observed_velocities_.clear();
    return JointTrajectoryController::on_deactivate(previous_state);
  }

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override
  {
    // The shared ownership decision can delay an update; observed timing is
    // telemetry, never a certified latency or physical stopping bound.
    std::lock_guard<std::mutex> lock(owner_mutex_);
    if (sequence_ > 0 && time.seconds() <= simulation_time_) {
      instance_id_ = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
      stationary_samples_ = 0;
      observation_.clear();
      observed_positions_.clear();
      if (contract_state_ != "unlocked") {request_stop("simulation_time_did_not_advance");}
    }
    if (stop_requested_) {
      read_state_from_state_interfaces(state_current_);
      if (state_current_.positions.size() == params_.joints.size() && finite(state_current_.positions)) {
        hold_positions_ = state_current_.positions;
      } else if (last_native_positions_.size() == params_.joints.size()) {
        hold_positions_ = last_native_positions_;
      }
      const auto goal = *rt_active_goal_.readFromRT();
      if (goal) {
        auto response = std::make_shared<FollowJTrajAction::Result>();
        response->error_code = FollowJTrajAction::Result::PATH_TOLERANCE_VIOLATED;
        response->error_string = fault_reason_;
        goal->setAborted(response);
        rt_active_goal_.writeFromNonRT(RealtimeGoalHandlePtr());
      }
      rt_has_pending_goal_ = false;
      auto hold = set_hold_position();
      // Invalid feedback must not become a new hardware command. The retained
      // finite command still gives no authority to assert that motion stopped.
      hold->points.front().positions = hold_positions_;
      add_new_trajectory_msg(hold);
      rt_is_holding_ = true;
      stop_requested_ = false;
    }
    const auto result = JointTrajectoryController::update(time, period);
    if (contract_state_ == "holding" || contract_state_ == "fault" || contract_state_ == "terminal") {
      // Retain one hold command; following a drifting observation would conceal it.
      for (size_t i = 0; i < hold_positions_.size(); ++i) {
        joint_command_interface_[0][i].get().set_value(hold_positions_[i]);
      }
    }
    if (contract_state_ != "unlocked") {
      std::vector<double> commands;
      for (const auto & command : joint_command_interface_[0]) {
        commands.push_back(command.get().get_value());
      }
      if (commands.size() == params_.joints.size() && finite(commands)) {
        last_native_positions_ = std::move(commands);
      }
    }
    if (contract_state_ == "active" && owned_goal_ && !owned_goal_->is_active()) {
      // The result remains owned by FollowJointTrajectory. Inactive alone does
      // not establish success or a physical stop; retain ownership and observe.
      request_stop("native_action_terminal_requires_observed_stop");
      contract_state_ = "terminal";
    }
    const auto command = *traj_msg_external_point_ptr_.readFromRT();
    if (command != last_command_) {last_command_ = command; ++command_revision_;}
    const auto & positions = state_current_.positions;
    const auto & velocities = state_current_.velocities;
    if (positions.size() != params_.joints.size() || !finite(positions) || !finite(velocities)) {
      observation_.clear();
      stationary_samples_ = 0;
      if (contract_state_ != "unlocked") {request_stop("nonfinite_owner_state");}
      return result;
    }
    const bool stopped = velocities.size() == positions.size() && positions == observed_positions_ &&
      std::all_of(velocities.begin(), velocities.end(), [](double value) {return value == 0.;});
    stationary_samples_ = stopped ? stationary_samples_ + 1 : 0;
    observed_positions_ = positions;
    observed_velocities_ = velocities;
    simulation_time_ = time.seconds();
    observation_time_ = std::chrono::steady_clock::now();
    maximum_period_ = std::max(maximum_period_, period.seconds());
    for (size_t i = 0; i < state_error_.positions.size() && i < maximum_error_.size(); ++i) {
      if (std::isfinite(state_error_.positions[i])) {
        maximum_error_[i] = std::max(maximum_error_[i], std::abs(state_error_.positions[i]));
      }
    }
    ++sequence_;
    publish_observation();
    return result;
  }

private:
  static bool finite(const std::vector<double> & values)
  {
    return std::all_of(values.begin(), values.end(), [](double value) {return std::isfinite(value);});
  }

  bool active() const
  {
    return get_node()->get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;
  }

  bool busy()
  {
    const auto goal = *rt_active_goal_.readFromNonRT();
    return (goal && goal->gh_ && goal->gh_->is_active()) || rt_has_pending_goal_.load() ||
           (owned_goal_ && (owned_goal_->is_active() || owned_goal_->is_canceling()));
  }

  bool observation_fresh(bool require_simulation_clock = true) const
  {
    if (observation_.empty()) {return false;}
    const auto wall_age = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - observation_time_).count();
    // These are observation availability cutoffs, never physical delay bounds.
    // A paused simulation clock cannot keep an old owner snapshot fresh.
    if (wall_age < 0. || wall_age > 2.) {return false;}
    if (!require_simulation_clock) {return true;}
    const auto age = get_node()->now().seconds() - simulation_time_;
    return std::isfinite(age) && age >= 0. && age <= 2.;
  }

  std::string validate_snapshot(const Contract::Request & request)
  {
    if (!active() || observation_.empty() || !get_node()->get_parameter("use_sim_time").as_bool()) {
      return "Active simulation owner observation unavailable";
    }
    if (request.expected_instance_id != instance_id_ ||
      request.expected_command_revision != command_revision_ ||
      request.expected_contract_revision != contract_revision_ ||
      *traj_msg_external_point_ptr_.readFromNonRT() != last_command_ ||
      request.expected_positions != observed_positions_)
    {
      return "Native owner snapshot changed";
    }
    if (!observation_fresh()) {return "Native owner observation is stale";}
    return "";
  }

  std::string validate_motion(const Contract::Request & request)
  {
    if (observed_velocities_.size() != params_.joints.size()) {
      return "Complete observed joint velocities unavailable";
    }
    if (busy() || stationary_samples_ < 2 || !rt_is_holding_.load()) {
      return "Native owner has not observed a stationary hold";
    }
    if (params_.command_interfaces != std::vector<std::string>{"position"} ||
      params_.interpolation_method != "splines")
    {
      return "Native ownership supports position commands with splines only";
    }
    if (request.stationary) {
      if (!request.trajectory.points.empty() || !request.trajectory.joint_names.empty()) {
        return "Stationary ownership cannot include motion";
      }
      return "";
    }
    const auto & trajectory = request.trajectory;
    if (trajectory.joint_names != params_.joints || trajectory.points.size() < 2 ||
      trajectory.header.stamp.sec != 0 || trajectory.header.stamp.nanosec != 0 ||
      !validate_trajectory_msg(trajectory) || trajectory.points.front().positions != observed_positions_ ||
      trajectory.points.front().time_from_start.sec != 0 || trajectory.points.front().time_from_start.nanosec != 0 ||
      std::none_of(request.controller_goal_id.begin(), request.controller_goal_id.end(),
        [](uint8_t value) {return value != 0;}))
    {
      return "Exact native trajectory, start or goal identity unavailable";
    }
    for (const auto & point : trajectory.points) {
      if (point.positions.size() != params_.joints.size() ||
        !finite(point.positions) || !finite(point.velocities) ||
        !finite(point.accelerations) || !finite(point.effort))
      {
        return "Native trajectory contains incomplete or nonfinite values";
      }
    }
    return "";
  }

  std::string apply_contract(const Contract::Request & request)
  {
    if (request.operation != "prepare" && request.operation != "arm" &&
      request.operation != "stop" && request.operation != "release")
    {
      return "Unknown native ownership operation";
    }
    if (request.operation == "stop") {
      // Stopping must remain available while positions and command revisions
      // change. Only the same active owner reservation may request it.
      if (!active() || request.expected_instance_id != instance_id_ ||
        contract_state_ == "unlocked" || request.reservation_token != reservation_token_ ||
        request.binding_fingerprint != binding_fingerprint_)
      {
        return "Native ownership identity changed";
      }
      request_stop("owner_requested_stop");
      return "";
    }
    const auto snapshot_error = validate_snapshot(request);
    if (!snapshot_error.empty()) {return snapshot_error;}
    if (request.operation == "prepare" || request.operation == "arm") {
      if (contract_state_ != "unlocked") {return "Native owner is already reserved";}
      const auto motion_error = validate_motion(request);
      if (!motion_error.empty() || request.operation == "prepare") {return motion_error;}
      if (request.reservation_token.empty() || request.binding_fingerprint.empty()) {
        return "Native ownership requires a reservation and physical binding";
      }
      reservation_token_ = request.reservation_token;
      binding_fingerprint_ = request.binding_fingerprint;
      trajectory_ = request.trajectory;
      controller_goal_id_ = request.controller_goal_id;
      stationary_ = request.stationary;
      hold_positions_ = observed_positions_;
      last_native_positions_ = hold_positions_;
      goal_claimed_ = false;
      owned_goal_.reset();
      fault_reason_.clear();
      maximum_period_ = 0.;
      maximum_error_.assign(params_.joints.size(), 0.);
      contract_state_ = stationary_ ? "holding" : "armed";
      ++contract_revision_;
      return "";
    }
    if (contract_state_ == "unlocked" || request.reservation_token != reservation_token_ ||
      request.binding_fingerprint != binding_fingerprint_)
    {
      return "Native ownership identity changed";
    }
    if (busy() || stop_requested_ || stationary_samples_ < 2 || contract_state_ == "active" ||
      (goal_claimed_ && contract_state_ == "armed"))
    {
      return "Native ownership retained until observed stationary completion";
    }
    contract_state_ = "unlocked";
    reservation_token_.clear();
    binding_fingerprint_.clear();
    trajectory_ = trajectory_msgs::msg::JointTrajectory();
    goal_claimed_ = stationary_ = false;
    owned_goal_.reset();
    ++contract_revision_;
    return "";
  }

  void request_stop(const std::string & reason)
  {
    // A repeated stop must not replace the first retained hold with drifted q.
    if (contract_state_ == "fault" || contract_state_ == "terminal") {return;}
    contract_state_ = "fault";
    fault_reason_ = reason;
    stop_requested_ = true;
    stationary_samples_ = 0;
    ++contract_revision_;
  }

  void fill_response(Contract::Response & response)
  {
    response.instance_id = instance_id_;
    response.command_revision = command_revision_;
    response.contract_revision = contract_revision_;
    response.reservation_token = reservation_token_;
    response.binding_fingerprint = binding_fingerprint_;
    response.contract_state = contract_state_;
    response.stationary = stationary_;
    response.observed_stationary = observation_fresh() && stationary_samples_ >= 2 &&
      !busy() && !stop_requested_;
    response.stationary_samples = stationary_samples_;
    response.rejected_commands = rejected_commands_;
    response.joint_names = params_.joints;
    response.observed_positions = observed_positions_;
    response.observed_velocities = observed_velocities_;
    response.simulation_time = simulation_time_;
    response.maximum_observed_update_period = maximum_period_;
    response.maximum_observed_position_error = maximum_error_;
    response.physical_execution_verified = false;
    response.physical_execution_reason = physical_reason_;
  }

  template<typename T>
  static void json_array(std::ostream & row, const std::vector<T> & values)
  {
    row << '[';
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) {row << ',';}
      row << values[i];
    }
    row << ']';
  }

  void publish_observation()
  {
    const auto goal = *rt_active_goal_.readFromNonRT();
    const bool has_active_goal = (goal && goal->gh_ && goal->gh_->is_active()) ||
      (owned_goal_ && (owned_goal_->is_active() || owned_goal_->is_canceling()));
    std::ostringstream row;
    row << std::setprecision(17)
        << "{\"version\":1,\"instance_id\":" << std::quoted(instance_id_)
        << ",\"controller\":" << std::quoted(get_node()->get_node_base_interface()->get_fully_qualified_name())
        << ",\"sequence\":" << sequence_ << ",\"command_revision\":" << command_revision_
        << ",\"simulation_time\":" << simulation_time_
        << ",\"has_active_goal\":" << (has_active_goal ? "true" : "false")
        << ",\"has_pending_goal\":" << (rt_has_pending_goal_.load() ? "true" : "false")
        << ",\"holding\":" << (rt_is_holding_.load() ? "true" : "false")
        << ",\"contract_revision\":" << contract_revision_
        << ",\"contract_state\":" << std::quoted(contract_state_)
        << ",\"reservation_token\":" << std::quoted(reservation_token_)
        << ",\"binding_fingerprint\":" << std::quoted(binding_fingerprint_)
        << ",\"fault_reason\":" << std::quoted(fault_reason_)
        << ",\"observed_stationary\":" << (stationary_samples_ >= 2 && !busy() && !stop_requested_ ? "true" : "false")
        << ",\"stationary_samples\":" << stationary_samples_
        << ",\"rejected_commands\":" << rejected_commands_
        << ",\"maximum_observed_update_period\":" << maximum_period_
        << ",\"physical_execution_verified\":false,\"physical_execution_reason\":" << std::quoted(physical_reason_)
        << ",\"joint_names\":[";
    for (size_t i = 0; i < params_.joints.size(); ++i) {
      if (i) {row << ',';}
      row << std::quoted(params_.joints[i]);
    }
    row << "],\"positions\":";
    json_array(row, observed_positions_);
    row << ",\"velocities\":";
    json_array(row, observed_velocities_);
    row << ",\"maximum_observed_position_error\":";
    json_array(row, maximum_error_);
    row << '}';
    observation_ = row.str();
  }

  std::mutex owner_mutex_;
  std::string observation_, instance_id_, reservation_token_, binding_fingerprint_, fault_reason_;
  std::string contract_state_{"unlocked"};
  uint64_t sequence_{0}, command_revision_{0}, contract_revision_{0}, rejected_commands_{0};
  uint64_t stationary_samples_{0};
  bool stationary_{false}, goal_claimed_{false}, stop_requested_{false};
  double simulation_time_{0.}, maximum_period_{0.};
  std::chrono::steady_clock::time_point observation_time_{};
  std::vector<double> observed_positions_, observed_velocities_, hold_positions_, maximum_error_;
  std::vector<double> last_native_positions_;
  trajectory_msgs::msg::JointTrajectory trajectory_;
  rclcpp_action::GoalUUID controller_goal_id_{};
  std::shared_ptr<GoalHandle> owned_goal_;
  std::shared_ptr<trajectory_msgs::msg::JointTrajectory> last_command_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr observation_service_;
  rclcpp::Service<Contract>::SharedPtr contract_service_;
};
}  // namespace cais_lab_robotics

PLUGINLIB_EXPORT_CLASS(
  cais_lab_robotics::RecoveryJointTrajectoryController,
  controller_interface::ControllerInterface)

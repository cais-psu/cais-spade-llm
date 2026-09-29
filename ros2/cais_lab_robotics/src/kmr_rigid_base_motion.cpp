// Keep parked kinematic KMR links rigid during the planar plugin's base rotation.
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Link.hh>
#include <gazebo/physics/Model.hh>

#include <cmath>

namespace gazebo
{
class KMRRigidBaseMotion : public ModelPlugin
{
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr) override
  {
    model_ = model;
    base_ = model_->GetLink("KMR_base_link");
    if (!base_)
    {
      gzerr << "KMR rigid base motion requires KMR_base_link\n";
      return;
    }
    update_ = event::Events::ConnectWorldUpdateBegin(
      [this](const common::UpdateInfo &) { Update(); });
  }

private:
  void Update()
  {
    const auto angular = base_->WorldAngularVel();
    const bool rotating = angular.Length() > 1e-9;
    if (!rotating && !was_rotating_)
    {
      return;
    }
    const auto origin = base_->WorldCoGPose().Pos();
    const auto linear = base_->WorldCoGLinearVel();
    // Model::SetLinearVel gives every link the same translation. Kinematic
    // links also need omega cross r to orbit the base during a yaw change.
    // Base execution permits this only with the arm in its observed park pose.
    for (const auto &link : model_->GetLinks())
    {
      if (link == base_)
      {
        continue;
      }
      const auto offset = link->WorldCoGPose().Pos() - origin;
      link->SetLinearVel(linear + angular.Cross(offset));
      link->SetAngularVel(angular);
    }
    was_rotating_ = rotating;
  }

  physics::ModelPtr model_;
  physics::LinkPtr base_;
  event::ConnectionPtr update_;
  bool was_rotating_ = false;
};
GZ_REGISTER_MODEL_PLUGIN(KMRRigidBaseMotion)
}

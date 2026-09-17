# Third-party assets in `KMR`

## LBR iiwa 14 R820

- Source: `ICube-Robotics/iiwa_ros2`
- Source URL: <https://github.com/ICube-Robotics/iiwa_ros2>
- Source commit: `9d048b901f8fc4acaa9a0ad3f52067bd4476093a`
- Vendored content: the visual and collision meshes referenced by
  `iiwa_description/urdf/iiwa.urdf.xacro` for `lbr_iiwa_14_r820`
- License: Apache License 2.0, copied to
  [`LICENSES/iiwa_ros2-Apache-2.0.txt`](LICENSES/iiwa_ros2-Apache-2.0.txt)

The source world retains the layout-only static model. The recovery launch now
uses the source revision's LBR iiwa joint transforms, limits, inertias, meshes,
and MoveIt naming in the separately spawned articulated simulation model at
`urdf/KMR_recovery.urdf.xacro`. The project-owned controller configuration uses
standard `gazebo_ros2_control` trajectory controllers.

## OnRobot RG2

- Source: `tony0404/OnRobot_ROS2_Description`
- Source URL: <https://github.com/tony0404/OnRobot_ROS2_Description>
- Source commit: `29180b3fa9cba6555f3e515e789b8ccd34252fab`
- Vendored content: RG2 `base_link`, `outer_knuckle`, `inner_knuckle`, and
  `inner_finger` visual and collision meshes
- License: MIT License, copied to
  [`LICENSES/OnRobot_ROS2_Description-MIT.txt`](LICENSES/OnRobot_ROS2_Description-MIT.txt)

The repeated left and right finger meshes and linkage reproduce the source RG2
xacro. The simulation exposes `KMR_rg2_finger_width` over the configured
0.015–0.110 m range. The nominal 20 mm adapter pose remains unverified against
the lab hardware.

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS 2 ${ROS_DISTRO} is not installed at ${ROS_SETUP}." >&2
  echo "Install ROS 2 first, then rerun this script." >&2
  exit 1
fi

mkdir -p "${ROS2_WS}/src"

if [[ ! -d "${ROS2_WS}/src/xarm_ros2/.git" ]]; then
  git clone -b "${ROS_DISTRO}" https://github.com/xArm-Developer/xarm_ros2.git --recursive "${ROS2_WS}/src/xarm_ros2"
else
  echo "xarm_ros2 already present at ${ROS2_WS}/src/xarm_ros2"
fi

if [[ ! -d "${ROS2_WS}/src/OnRobot_ROS2_Description/.git" ]]; then
  git clone https://github.com/tonydle/OnRobot_ROS2_Description.git "${ROS2_WS}/src/OnRobot_ROS2_Description"
else
  echo "OnRobot_ROS2_Description already present at ${ROS2_WS}/src/OnRobot_ROS2_Description"
fi

mkdir -p "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/worlds"
mkdir -p "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/launch"

cp "${REPO_ROOT}/ros2/cais_lab_gazebo/worlds/table.world" \
  "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/worlds/table.world"
cp "${REPO_ROOT}/ros2/cais_lab_gazebo/launch/xarm6_ur5e_gazebo.launch.py" \
  "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/launch/xarm6_ur5e_gazebo.launch.py"

source "${ROS_SETUP}"
cd "${ROS2_WS}"
colcon build

cat <<EOF

Gazebo workspace is ready.

Next steps:
  source ${ROS_SETUP}
  source ${ROS2_WS}/install/setup.bash
  ros2 launch xarm_gazebo xarm6_ur5e_gazebo.launch.py

Then, in a separate terminal:
  cd ${REPO_ROOT}
  poetry run python -m cais_spade_llm
EOF

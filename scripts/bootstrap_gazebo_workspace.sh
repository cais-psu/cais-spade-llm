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

if [[ ! -d "${ROS2_WS}/src/IFRA_LinkAttacher/.git" ]]; then
  git clone https://github.com/IFRA-Cranfield/IFRA_LinkAttacher.git "${ROS2_WS}/src/IFRA_LinkAttacher"
else
  echo "IFRA_LinkAttacher already present at ${ROS2_WS}/src/IFRA_LinkAttacher"
fi

copy_file_if_not_same() {
  local src="$1"
  local dst_dir="$2"
  local dst="${dst_dir}/$(basename "${src}")"
  if [[ -e "${dst}" ]] && [[ "$(readlink -f "${src}")" == "$(readlink -f "${dst}")" ]]; then
    return
  fi
  if [[ -L "${dst}" ]]; then
    rm -f "${dst}"
  fi
  cp "${src}" "${dst_dir}/"
}

copy_glob_if_not_same() {
  local src_dir="$1"
  local pattern="$2"
  local dst_dir="$3"
  local src
  shopt -s nullglob
  for src in "${src_dir}"/${pattern}; do
    copy_file_if_not_same "${src}" "${dst_dir}"
  done
  shopt -u nullglob
}

copy_tree_if_not_same() {
  local src_dir="$1"
  local dst_dir="$2"
  local src rel rel_dir
  if [[ ! -d "${src_dir}" ]]; then
    return
  fi
  while IFS= read -r -d '' src; do
    rel="${src#${src_dir}/}"
    rel_dir="$(dirname "${rel}")"
    mkdir -p "${dst_dir}/${rel_dir}"
    copy_file_if_not_same "${src}" "${dst_dir}/${rel_dir}"
  done < <(find "${src_dir}" -type f -print0)
}

remove_legacy_cais_config_files() {
  local config_dir="$1"
  rm -f \
    "${config_dir}/hardware_arms.yaml" \
    "${config_dir}/xarm6_ur5e_controllers.yaml" \
    "${config_dir}/ur5e_rg2_controllers.yaml" \
    "${config_dir}/ur5e_initial_positions.yaml"
}

mkdir -p "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/worlds"
mkdir -p "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/launch"
mkdir -p "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/config"
mkdir -p "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/rviz"

copy_glob_if_not_same "${REPO_ROOT}/ros2/cais_lab_robotics/worlds" "*.world" \
  "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/worlds"
copy_glob_if_not_same "${REPO_ROOT}/ros2/cais_lab_robotics/launch" "*.py" \
  "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/launch"
copy_tree_if_not_same "${REPO_ROOT}/ros2/cais_lab_robotics/config" \
  "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/config"
remove_legacy_cais_config_files "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/config"
copy_glob_if_not_same "${REPO_ROOT}/ros2/cais_lab_robotics/rviz" "*.rviz" \
  "${ROS2_WS}/src/xarm_ros2/xarm_gazebo/rviz"
copy_file_if_not_same \
  "${REPO_ROOT}/ros2/third_party/IFRA_LinkAttacher/ros2_LinkAttacher/src/gazebo_link_attacher.cpp" \
  "${ROS2_WS}/src/IFRA_LinkAttacher/ros2_LinkAttacher/src"

# ROS setup scripts are not consistently safe under `set -u`.
set +u
source "${ROS_SETUP}"
set -u

cd "${ROS2_WS}"
# Skip optional xArm vision/hand-eye package that pulls in
# object_recognition_msgs, which is not needed for this repo's dual-robot
# Gazebo + MoveIt bring-up.
colcon build --packages-skip d435i_xarm_setup

# Upstream xarm_gazebo installs only `worlds/` and `launch/`. Our custom dual-
# robot flow also requires `config/` and `rviz/` assets at runtime.
mkdir -p "${ROS2_WS}/install/xarm_gazebo/share/xarm_gazebo/config"
mkdir -p "${ROS2_WS}/install/xarm_gazebo/share/xarm_gazebo/rviz"
copy_tree_if_not_same "${REPO_ROOT}/ros2/cais_lab_robotics/config" \
  "${ROS2_WS}/install/xarm_gazebo/share/xarm_gazebo/config"
remove_legacy_cais_config_files "${ROS2_WS}/install/xarm_gazebo/share/xarm_gazebo/config"
copy_glob_if_not_same "${REPO_ROOT}/ros2/cais_lab_robotics/rviz" "*.rviz" \
  "${ROS2_WS}/install/xarm_gazebo/share/xarm_gazebo/rviz"

cat <<EOF

Gazebo workspace is ready.

Next steps:
  source ${ROS_SETUP}
  source ${ROS2_WS}/install/setup.bash
  ros2 launch xarm_gazebo dual_moveit_gazebo.launch.py

Then, in a separate terminal:
  cd ${REPO_ROOT}
  poetry run python -m cais_spade_llm
EOF

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
CAIS_PACKAGE_SOURCE="${REPO_ROOT}/ros2/cais_lab_robotics"
CAIS_PACKAGE_LINK="${ROS2_WS}/src/cais_lab_robotics"

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

if [[ ! -f "${CAIS_PACKAGE_SOURCE}/package.xml" ]]; then
  echo "cais_lab_robotics package is missing at ${CAIS_PACKAGE_SOURCE}." >&2
  exit 1
fi

if [[ -L "${CAIS_PACKAGE_LINK}" ]]; then
  if [[ "$(readlink -f "${CAIS_PACKAGE_LINK}")" != "$(readlink -f "${CAIS_PACKAGE_SOURCE}")" ]]; then
    echo "${CAIS_PACKAGE_LINK} points to a different package. Remove it manually and rerun." >&2
    exit 1
  fi
elif [[ -e "${CAIS_PACKAGE_LINK}" ]]; then
  echo "${CAIS_PACKAGE_LINK} already exists and is not the CAIS repository symlink." >&2
  echo "Move it out of the way manually, then rerun this script." >&2
  exit 1
else
  ln -s "${CAIS_PACKAGE_SOURCE}" "${CAIS_PACKAGE_LINK}"
fi

# colcon does not remove deleted package assets from an existing install tree.
STALE_FAST_WORLD="${ROS2_WS}/install/cais_lab_robotics/share/cais_lab_robotics/worlds/table_fast.world"
if [[ -e "${STALE_FAST_WORLD}" || -L "${STALE_FAST_WORLD}" ]]; then
  rm -f "${STALE_FAST_WORLD}"
fi

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
colcon build --executor sequential --packages-skip d435i_xarm_setup

cat <<EOF

Gazebo workspace is ready.

Next steps:
  source ${ROS_SETUP}
  source ${ROS2_WS}/install/setup.bash
  ros2 launch cais_lab_robotics dual_moveit_gazebo.launch.py

Then, in a separate terminal:
  cd ${REPO_ROOT}
  poetry run python -m cais_spade_llm
EOF

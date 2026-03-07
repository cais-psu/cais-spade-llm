#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
ROS_SETUP="/opt/ros/humble/setup.bash"
ROS2_WS_SETUP="${HOME}/ros2_ws/install/setup.bash"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv interpreter at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -f "${ROS_SETUP}" ]]; then
  set +u
  source "${ROS_SETUP}"
  set -u
fi

if [[ -f "${ROS2_WS_SETUP}" ]]; then
  set +u
  source "${ROS2_WS_SETUP}"
  set -u
fi

exec "${PYTHON_BIN}" "$@"

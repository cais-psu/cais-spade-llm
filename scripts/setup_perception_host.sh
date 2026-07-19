#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
TARGET_USER="${SUDO_USER:-${USER}}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this one-time provisioning script with sudo:" >&2
  echo "  sudo bash scripts/setup_perception_host.sh" >&2
  echo "The script does not store the sudo password or change sudoers." >&2
  exit 1
fi

if ! id "${TARGET_USER}" >/dev/null 2>&1; then
  echo "Target operator user does not exist: ${TARGET_USER}" >&2
  exit 1
fi

apt-get update
apt-get install -y \
  "ros-${ROS_DISTRO}-realsense2-camera" \
  "ros-${ROS_DISTRO}-realsense2-description" \
  "ros-${ROS_DISTRO}-rqt-image-view" \
  usbutils \
  v4l-utils

getent group video >/dev/null || groupadd video
usermod -aG video "${TARGET_USER}"

# The ROS librealsense package installs the RealSense device rules when they
# are available for this platform. Reload them on native Linux; WSL uses
# usbipd-win attachment and creates /dev/video* through the WSL kernel.
if command -v udevadm >/dev/null 2>&1; then
  udevadm control --reload-rules || true
  udevadm trigger || true
fi

cat <<EOF

Perception host provisioning is complete for ${TARGET_USER}.

1. Sign out and back in (or restart WSL) so video-group membership is active.
2. On WSL, bind each RealSense once from Administrator PowerShell:
     usbipd list
     usbipd bind --busid <BUSID>
3. Open the CAIS UI and use Perception -> Attach to WSL after each unplug or WSL restart.

No sudo password was stored and no passwordless sudo rule was added.
EOF

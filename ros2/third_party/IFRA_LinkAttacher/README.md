# IFRA LinkAttacher (Patched Source Mirror)

This folder mirrors the patched IFRA source file used by this project:

- `ros2_LinkAttacher/src/gazebo_link_attacher.cpp`

Why this mirror exists:

- Keep all local modifications under this repository (`ros2/`) for GitHub tracking.
- Reapply deterministic plugin behavior on a fresh machine by copying this file into:
  `~/ros2_ws/src/IFRA_LinkAttacher/ros2_LinkAttacher/src/`

Patched behavior in this mirrored source:

- Allows simultaneous attachments (multiple robot grasps in one run).
- Prevents a single link from being attached to multiple targets at once, with
  an explicit exception for `assembly_board_v1::link` so the carrier retains
  both fixtures and all eleven assembled components.
- Preserves current relative grasp pose at attach time (avoids hard snap to link origin).
- Runs joint attachment and removal on the Gazebo physics update thread. ROS service
  responses follow that update; pending requests time out without a late mutation.

After copying, rebuild:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select linkattacher_msgs ros2_linkattacher xarm_gazebo
```

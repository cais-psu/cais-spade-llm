import time

from rtde_control import RTDEControlInterface
from scipy.spatial.transform import Rotation


ROBOT_IP = "192.168.1.172"


class RTDECommander:
    def __init__(self, robot_state=None, hostname=ROBOT_IP, speed=0.3, acceleration=0.5):
        self.robot_state = robot_state
        self.rtde = RTDEControlInterface(hostname=hostname)
        self.speed = speed
        self.acceleration = acceleration

    def build_pose(self, pose_mm_rpy):
        x, y, z, roll, pitch, yaw = [float(value) for value in pose_mm_rpy]
        rotvec = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_rotvec()
        return [x / 1000.0, y / 1000.0, z / 1000.0, *rotvec]

    def move_to_pose(self, pose_mm_rpy, blocking=True):
        self.rtde.moveL(
            self.build_pose(pose_mm_rpy),
            self.speed,
            self.acceleration,
            asynchronous=not blocking,
        )

    def move_to_cartesian(self, x, y, z, roll=-180, pitch=0, yaw=0, blocking=True):
        self.move_to_pose([x, y, z, roll, pitch, yaw], blocking=blocking)

    def _set_gripper(self, width_mm, force, settle_s, blocking=True):
        body = f"""
      local rg = rpc_factory("xmlrpc","http://localhost:41414")
      local ret = rg.rg_grip(0, {float(width_mm)}, {float(force)})
      textmsg("rg_grip returned: ", ret)
    """
        self.rtde.sendCustomScriptFunction("rg2_cmd", body)
        if blocking:
            time.sleep(settle_s)

    def open_gripper(self, width_mm=70, force=10, blocking=True):
        self._set_gripper(width_mm, force, settle_s=1.2, blocking=blocking)

    def close_gripper(self, width_mm=10, force=40, blocking=True):
        self._set_gripper(width_mm, force, settle_s=2.0, blocking=blocking)

    def pick(self, pick_pose_mm_rpy, approach_mm=80):
        above_pick = self._above(pick_pose_mm_rpy, approach_mm)
        self.move_to_pose(above_pick)
        self.open_gripper()
        self.move_to_pose(pick_pose_mm_rpy)
        self.close_gripper()
        self.move_to_pose(above_pick)

    def place(self, place_pose_mm_rpy, approach_mm=80):
        above_place = self._above(place_pose_mm_rpy, approach_mm)
        self.move_to_pose(above_place)
        self.move_to_pose(place_pose_mm_rpy)
        self.open_gripper()
        self.move_to_pose(above_place)

    def pick_and_place(self, pick_pose_mm_rpy, place_pose_mm_rpy, approach_mm=80):
        self.pick(pick_pose_mm_rpy, approach_mm)
        self.place(place_pose_mm_rpy, approach_mm)

    def disconnect(self):
        self.rtde.disconnect()

    set_gripper_open = open_gripper
    set_gripper_close = close_gripper

    @staticmethod
    def _above(pose_mm_rpy, dz_mm):
        pose = list(pose_mm_rpy)
        if len(pose) != 6:
            raise ValueError("Pose must be [x, y, z, roll, pitch, yaw].")
        pose[2] += dz_mm
        return pose


if __name__ == "__main__":
    pick_pose = [-128.47, -510.42, 27.90, -180, 0, 0]
    place_pose = [-100.00, -200.00, 60.00, -180, 0, 0]

    robot = RTDECommander()
    try:
        robot.pick_and_place(pick_pose, place_pose)
    finally:
        robot.disconnect()

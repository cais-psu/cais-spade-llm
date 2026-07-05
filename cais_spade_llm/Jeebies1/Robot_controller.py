import time

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from scipy.spatial.transform import Rotation

try:
    from cais_spade_llm.Jeebies1.xarmlib.wrapper import XArmAPI
except ModuleNotFoundError as exc:
    if exc.name != "cais_spade_llm":
        raise
    from xarmlib.wrapper import XArmAPI


class UR5eRTDECommander:
    def __init__(self):
        self.hostname = "192.168.1.172"
        print(f"Connecting RTDE control/receive to {self.hostname}...")
        self.rtde = RTDEControlInterface(hostname=self.hostname)
        self.rtde_receive = RTDEReceiveInterface(hostname=self.hostname)
        print("RTDE connected.")
        self.speed = 0.5
        self.acceleration = 0.3
        self.roll = -180
        self.pitch = 0
        self.yaw = 0
        self.approach_offset = 80

    def build_pose(self, pose):
        x, y, z = [float(value) for value in pose]
        rotvec = Rotation.from_euler(
            "xyz", [self.roll, self.pitch, self.yaw], degrees=True
        ).as_rotvec()
        return [x / 1000.0, y / 1000.0, z / 1000.0, *rotvec]

    def move_to_pose(self, pose):
        self.rtde.moveL(self.build_pose(pose), self.speed, self.acceleration)

    def move_tcp_z(self, delta_mm=10.0):
        start_pose = list(self.rtde_receive.getActualTCPPose())
        target_pose = list(start_pose)
        target_pose[2] += float(delta_mm) / 1000.0
        print(f"Current TCP pose: {start_pose}")
        print(f"Target TCP pose:  {target_pose}")
        result = self.rtde.moveL(target_pose, 0.05, 0.1, asynchronous=False)
        time.sleep(0.2)
        end_pose = list(self.rtde_receive.getActualTCPPose())
        actual_delta_mm = (end_pose[2] - start_pose[2]) * 1000.0
        print(f"moveL result: {result}")
        print(f"End TCP pose:    {end_pose}")
        print(f"Actual z delta:  {actual_delta_mm:.3f} mm")
        print(f"RTDE program running: {self.rtde.isProgramRunning()}")

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

    def disconnect(self):
        self.rtde_receive.disconnect()
        self.rtde.disconnect()

    def build_intermediate_pose(self, pose):
        pose[2] += self.approach_offset
        return pose

    def go_home(self):
        p = [-100, -200, 150, -180, 0, 0]
        pose = self.build_pose(p)
        self.rtde.moveL(pose, self.speed, self.acceleration, asynchronous=False)

    def pick(self, pose):
        intermediate_pose = self.build_intermediate_pose(pose)
        self.rtde.moveL(self.build_pose(intermediate_pose), self.speed, self.acceleration)
        self.open_gripper()
        self.rtde.moveL(self.build_pose(pose), self.speed, self.acceleration)
        self.close_gripper()
        self.rtde.moveL(self.build_pose(intermediate_pose), self.speed, self.acceleration)

    def place(self, pose):
        intermediate_pose = self.build_intermediate_pose(pose)
        self.rtde.moveL(self.build_pose(intermediate_pose), self.speed, self.acceleration)
        self.rtde.moveL(self.build_pose(pose), self.speed, self.acceleration)
        self.open_gripper()
        self.rtde.moveL(self.build_pose(intermediate_pose), self.speed, self.acceleration)


class xArmCommander:
    def __init__(self):
        self.arm = XArmAPI("192.168.1.240")
        self.speed = 280
        self.acceleration = 10000
        self.roll = 180
        self.pitch = 0
        self.yaw = 0
        self.approach_offset = 80
        self.radius = -1
        self.wait = True
        self.grip_speed = 1000
        self.gripper_open_position = 850
        self.gripper_close_position = 0

    def move_to_pose(self, pose):
        code = self.arm.set_position(
            *pose,
            speed=self.speed,
            mvacc=self.acceleration,
            radius=self.radius,
            wait=self.wait,
        )

    def close_gripper(self):
        self.arm.set_gripper_position(self.gripper_close_position, speed=self.grip_speed)

    def open_gripper(self):
        self.arm.gripper_open_position(self.gripper_close_position, speed=self.grip_speed)

    def build_intermediate_pose(self, pose):
        pose[2] += self.approach_offset
        return pose

    def go_home(self):
        self.move_to_pose([250, -150, 445, 180, 0, 0])

    def pick(self, pose):
        intermediate_pose = self.build_intermediate_pose(pose)
        self.move_to_pose(intermediate_pose)
        self.open_gripper()
        self.move_to_pose(pose)
        self.close_gripper()
        self.move_to_pose(intermediate_pose)

    def place(self, pose):
        intermediate_pose = self.build_intermediate_pose(pose)
        self.move_to_pose(intermediate_pose)
        self.move_to_pose(pose)
        self.open_gripper()
        self.move_to_pose(intermediate_pose)


if __name__ == "__main__":
    ur5e_commander = UR5eRTDECommander()

    try:
        ur5e_commander.move_tcp_z(-100.0)
    finally:
        ur5e_commander.disconnect()

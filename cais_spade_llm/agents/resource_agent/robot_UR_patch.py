# urx_patches.py

import urx

def apply_urx_patches():
    # -------------------------
    # Patch robot.py: Robot.getl
    # -------------------------
    Robot = urx.robot.Robot
    _orig_get_pose = Robot.get_pose

    def getl(self, wait=False, _log=True):
        """
        return current transformation from tcp to current csys
        """
        t = _orig_get_pose(self, wait, _log)

        # Your requested behavior:
        # return t.pose_vector.get_array().tolist()
        # But different urx versions store pose differently, so we make it robust.
        pv = getattr(t, "pose_vector", None)
        if pv is not None:
            # Some versions: pv.get_array()
            if hasattr(pv, "get_array"):
                arr = pv.get_array()
                return arr.tolist() if hasattr(arr, "tolist") else list(arr)

            # Some versions: pv.array
            if hasattr(pv, "array"):
                arr = pv.array
                return arr.tolist() if hasattr(arr, "tolist") else list(arr)

        # Fallback: if t is already list-like / numpy-like
        if hasattr(t, "tolist"):
            return t.tolist()
        if isinstance(t, (list, tuple)):
            return list(t)

        raise TypeError(f"Don't know how to convert pose to list. pose type={type(t)}")

    Robot.getl = getl

    # ----------------------------
    # Patch urrobot.py: URRobot.movex
    # ----------------------------
    URRobot = urx.urrobot.URRobot

    def movex(self, command, tpose, acc=0.01, vel=0.01, wait=True,
              relative=False, threshold=None):
        """
        Send a move command to the robot.

        Since UR robots have several methods, this sends whatever is defined in
        'command' string (e.g., "movel", "movep").
        """

        # Convert PoseVector to array if needed
        if hasattr(tpose, "array"):
            tpose = tpose.array

        # Ensure python list
        if hasattr(tpose, "tolist"):
            tpose = tpose.tolist()
        else:
            tpose = list(tpose)

        # Handle relative motion
        if relative:
            l = self.getl()
            tpose = [v + l[i] for i, v in enumerate(tpose)]

        # Format and send the URScript command
        prog = self._format_move(command, tpose, acc, vel, prefix="p")
        self.send_program(prog)

        # Optionally wait until the motion is complete
        if wait:
            self._wait_for_move(tpose[:6], threshold=threshold)

        return self.getl()

    URRobot.movex = movex
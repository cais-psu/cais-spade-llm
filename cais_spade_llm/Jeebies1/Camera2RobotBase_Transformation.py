import numpy as np


class Cam2BaseUR5e:
    def __init__(self):
        """these camera intrinsics are calibrated with MATLAB repo "Estimate Pose of Moving Camera Mounted on a Robot"""
        self.fx = 623.9816462749620 # focal length of the camera in x
        self.fy = 613.8080113982506 # focal length of the camera in y
        self.cx = 318.5163260449835 # center point of the camera in x
        self.cy = 237.5512378918142 # center point of the camera in y

    def pixel_to_camera(self,u, v, z):
        """
        Convert pixel coordinates to 3D point in camera frame.
        """
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return x, y

    @staticmethod
    def camera_to_ee():
        """
        Transform point from camera frame to robot base frame.
        Note that this transformation matrix is calibrated with MATLAB repo "Estimate Pose of Moving Camera Mounted on a Robot"
        """
        return np.array([
        [-0.994932591781407, -0.095336836012590, 0.031937838221171, 25.592566657792],
        [0.096220059168056, -0.994983958303154, 0.027360974636959, 73.297417454375],
        [0.029169127940838, 0.030295386092556, 0.999115284417506, 0.08454406256067266],
        [0,0,0,1]])

    def get_coordinate(self, u, v, z, T):
        """
        Return the coordinate of the detected objects in the robot base frame
        """
        x, y = self.pixel_to_camera(u, v, z)  # real distance between center of the camera and detected object
        T_base_camera = T @ self.camera_to_ee
        point_camera = np.array([x, y, z, 1])  # in homogeneous form
        coordinate = T_base_camera @ point_camera
        return coordinate[:2]



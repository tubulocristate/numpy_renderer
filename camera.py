import numpy as np
import sys

class Camera():
    def __init__(self, camera_position, at, up, fov, z_near, z_far):
        self.camera_position = camera_position
        self.at = at
        self.up = up
        self.fov = fov
        self.z_near = z_near
        self.z_far = z_far

    def World2CameraMatrix(self, camera_position, at, up):
        camera_direction = camera_position - at
        camera_direction = camera_direction / np.linalg.norm(camera_direction)
        camera_right = np.cross(up, camera_direction)
        camera_right = camera_right / np.linalg.norm(camera_right)
        camera_up = np.cross(camera_direction, camera_right)

        R = np.eye(4)
        R[0, :3] = camera_right
        R[1, :3] = camera_up
        R[2, :3] = camera_direction

        T = np.eye(4)

        T[:3, -1] = -camera_position

        return R@T

    def Camera2ScreenMatrix(self, fov, z_near, z_far):
        projection_matrix = np.zeros(shape=(4, 4))
        projection_matrix[0][0] = 1 / (np.tan(fov / 2 * np.pi / 180))
        projection_matrix[1][1] = 1 / (np.tan(fov / 2 * np.pi / 180))
        projection_matrix[2][2] = (z_far + z_near) / (z_near - z_far)
        projection_matrix[2][3] = 2*z_far * z_near / (z_near - z_far)
        projection_matrix[3][2] = -1
        return projection_matrix

    def __call__(self, points):
        C2SM = self.Camera2ScreenMatrix(self.fov, self.z_near, self.z_far)
        W2CM = self.World2CameraMatrix(self.camera_position, self.at, self.up)
        projected_points = C2SM @ W2CM @ points.T
        projected_points[:-1] /= projected_points[-1]
        return projected_points.T
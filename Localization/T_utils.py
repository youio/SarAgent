import numpy as np
import cv2 as cv
import carla
import apriltag
import json
from cam_poses import T_camera_rover

class AprilTagLocalizer:
    def __init__(self, geometry_file, cam_matrix, tag_sizes, dist_coeffs=None):
        self.detector = apriltag.Detector()
        self.camera_matrix = cam_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(4)
        self.tag_sizes = tag_sizes
        self.tag_poses = self._load_tag_poses(geometry_file)

    def _load_tag_poses(self, geometry_file):
        tag_poses = {}
        with open(geometry_file, 'r') as f:
            data = json.load(f)
        
        for group, tags in data["fiducial_groups"].items():
            for tag_position, tag_data in tags.items():
                tag_id = tag_data["tag_id"]
                position = np.array(tag_data["position"])
                roll, pitch, yaw = tag_data["orientation"].values()

                R_tag_lander = self.euler_to_matrix(roll, pitch, yaw) # computing rotation Ms. from Euler Angles (Assuming ZYX order)
                T_tag_lander = np.eye(4)
                T_tag_lander[:3, :3] = R_tag_lander
                T_tag_lander[:3, 3] = position

                tag_poses[tag_id] = T_tag_lander  # Store transformation matrix

        return tag_poses
    
    def detect_tags(self, image):
        return self.detector.detect(image)
    
    def estimate_pose(self, tag_id, corners):

        if tag_id not in self.tag_sizes:
            print(f"Unknown tag ID: {tag_id}, skipping pose estimation.")
            return None

        tag_size = self.tag_sizes[tag_id]
        object_points = np.array([
            [-tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, -tag_size / 2, 0],
            [tag_size / 2, tag_size / 2, 0],
            [-tag_size / 2, tag_size / 2, 0]
        ], dtype=np.float32)

        image_points = np.array(corners, dtype=np.float32)
        ret, rvec, tvec = cv.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs)

        if not ret:
            print(f"Failed to estimate pose for tag {tag_id}")
            return None

        R, _ = cv.Rodrigues(rvec)
        T_tag_camera = np.eye(4)
        T_tag_camera[:3, :3] = R
        T_tag_camera[:3, 3] = tvec.ravel()

        return T_tag_camera
    
    def get_rover_pose(self, tag_id, T_tag_camera, T_camera_rover):
        
        if tag_id not in self.tag_poses:
            print(f"Tag {tag_id} not found in geometry file.")
            return None

        T_tag_lander = self.tag_poses[tag_id] # tag pose wrt lander
        T_rover_lander = T_tag_lander @ np.linalg.inv(T_tag_camera @ T_camera_rover)

        return T_rover_lander
    
if __name__ == "__main__":
    geometry_file = "cam_geometry.json"
    camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])
    dist_coeffs = np.zeros(4)
    tag_sizes = {243: 0.339, 71: 0.339, 462: 0.339, 37: 0.339, 0: 0.339, 3: 0.339, 2: 0.339, 1: 0.339,
                 10: 0.339, 11: 0.339, 8: 0.339, 9: 0.339, 464: 0.339, 459: 0.339, 258: 0.339, 5: 0.339}

    localizer = AprilTagLocalizer(geometry_file, camera_matrix, tag_sizes, dist_coeffs)
    # image = cv.imread("test_image.png", cv.IMREAD_GRAYSCALE)
    detected_tags = localizer.detect_tags(image)

    for tag in detected_tags:
        tag_id = tag.tag_id
        T_tag_camera = localizer.estimate_pose(tag_id, tag.corners)

        if T_tag_camera is not None:
            T_rover_lander = localizer.get_rover_pose(tag_id, T_tag_camera, T_camera_rover)
            print(f"Rover Pose wrt Lander (T_rover^lander):\n{T_rover_lander}")
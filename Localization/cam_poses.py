import numpy as np

def euler_to_matrix(roll, pitch, yaw):
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    return Rz @ Ry @ Rx

camera_poses = {
    "front_left": {
        "position": np.array([0.28, 0.081, 0.131]),
        "orientation": np.array([0, 0, 0])  # (roll, pitch, yaw)
    },
    "front_right": {
        "position": np.array([0.28, -0.081, 0.131]),
        "orientation": np.array([0, 0, 0])  # (roll, pitch, yaw)
    },
    "rear_left": {
        "position": np.array([-0.28, -0.081, 0.131]),
        "orientation": np.array([0, 0, np.pi])  # (roll, pitch, yaw)
    },
    "rear_right": {
        "position": np.array([-0.28, 0.081, 0.131]),
        "orientation": np.array([0, 0, np.pi])  # (roll, pitch, yaw)
    },
    "mid_left": {
        "position": np.array([0.015, 0.252, 0.132]),
        "orientation": np.array([0, 0, np.pi/2])  # (roll, pitch, yaw)
    },
    "mid_right": {
        "position": np.array([-0.015, -0.252, 0.132]),
        "orientation": np.array([0, 0, -np.pi/2])  # (roll, pitch, yaw)
    },
    "front_arm": {
        "position": np.array([0.414, 0.015, -0.038]),
        "orientation": np.array([0, 0, -np.pi/2])  # (roll, pitch, yaw)
    }
}

def T_camera_rover(camera_name):
    
    if camera_name not in camera_poses:
        raise ValueError(f"Unknown camera name: {camera_name}")

    position = camera_poses[camera_name]["position"]
    roll, pitch, yaw = camera_poses[camera_name]["orientation"]

    R = euler_to_matrix(roll, pitch, yaw)
    T_camera_rover = np.eye(4)
    T_camera_rover[:3, :3] = R
    T_camera_rover[:3, 3] = position

    return T_camera_rover


# if __name__ == "__main__":
#     camera_name = "front_left"  # Example
#     T = T_camera_rover(camera_name)
#     print(f"T_{camera_name}^rover:\n{T}")

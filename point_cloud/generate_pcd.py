import os

import cv2
# import time
# import argparse
# import glob
import numpy as np
# import torch
# from tqdm import tqdm
# import open3d as o3d

from utils import get_calibration_parameters, calc_depth_map, find_distances, add_depth, Open3dVisualizer, write_ply
# from object_detector import ObjectDetectorAPI
from disparity_estimator.raftstereo_disparity_estimator import RAFTStereoEstimator
import vizval
from ultralytics import YOLO
apollobot = YOLO("best.pt")

class generate_pcd:
    def __init__(self):
        self.disp_estimator = RAFTStereoEstimator()

    def get_pcd(self, left_image, right_image):
        img = left_image

        disparity_map = self.disp_estimator.estimate(left_image, right_image)

        disparity_left = disparity_map

        k_left = np.array([
            [1047.19,    0.0,   640.0],
            [   0.0,  1047.19,  360.0],
            [   0.0,     0.0,     1.0]
        ])
        k_right = k_left
        # t_left = np.array([[0.0], [0.0], [0.0]])  # Left camera translation vector
        # t_right = np.array([[-0.162], [0.0], [0.0]]) 
        # p_left = np.array([
        #     [1047.19,    0.0,   640.0,   0.0],
        #     [   0.0,  1047.19,  360.0,   0.0],
        #     [   0.0,     0.0,     1.0,   0.0]
        # ])

        # p_right = np.array([
        #     [1047.19,    0.0,   640.0, -169.64],
        #     [   0.0,  1047.19,  360.0,    0.0],
        #     [   0.0,     0.0,     1.0,    0.0]
        # ])

        # depth_map = calc_depth_map(disparity_map, k_left, t_left, t_right)
        disparity_map = (disparity_map * 256.).astype(np.uint16)
        color_depth = cv2.applyColorMap(cv2.convertScaleAbs(disparity_map, alpha=0.01), cv2.COLORMAP_JET)
        h = img.shape[0]
        w = img.shape[1]
        color_depth = cv2.resize(color_depth, (w, h))        
        # Calculate depth-to-disparity
        cam1 = k_left  # left image - P2
        cam2 = k_right  # right image - P3

        Tmat = np.array([0.54, 0., 0.])
        Q = np.zeros((4, 4))
        cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2,
                            distCoeffs1=0, distCoeffs2=0,
                            imageSize=img.shape[:2],
                            R=np.identity(3), T=Tmat,
                            R1=None, R2=None,
                            P1=None, P2=None, Q=Q)

        points = cv2.reprojectImageTo3D(disparity_left.copy(), Q)
        # reflect on x axis

        reflect_matrix = np.identity(3)
        reflect_matrix[0] *= -1
        points = np.matmul(points, reflect_matrix)

        img_left = left_image
        colors = cv2.cvtColor(img_left.copy(), cv2.COLOR_BGR2RGB)
        disparity_left = cv2.resize(disparity_left, (colors.shape[1], colors.shape[0]))
        points = cv2.resize(points, (colors.shape[1], colors.shape[0]))

        mask = disparity_left > disparity_left.min()
        out_points = points[mask]
        # print(out_points)
        out_colors = colors[mask]

        # Filter by depth range
        # min_depth = -20.0  # Set your minimum depth
        # max_depth = -5.0  # Set your maximum depth
        # depth_mask = (out_points[:, 2] >= min_depth) & (out_points[:, 2] <= max_depth)
        # out_points = out_points[depth_mask]
        # out_colors = out_colors[depth_mask]

        rock_mask = vizval.get_visual_prediction(apollobot, img_left)

        rock_mask_resized = cv2.resize(rock_mask, (colors.shape[1], colors.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Flatten and apply the mask
        rock_indices = rock_mask_resized[mask]
        # rock_indices = rock_indices[depth_mask]
        out_colors[rock_indices > 0] = [255, 0, 0]

        out_colors = out_colors.reshape(-1, 3)

        path_ply = os.path.join("output/ply/",)
        isExist = os.path.exists(path_ply)
        if not isExist:
            os.makedirs(path_ply)
        print("path_ply: {}".format(path_ply))

        file_name = path_ply + "/" +str(6) + ".ply"
        print("file_name: {}".format(file_name))
        write_ply(file_name, out_points, out_colors)

        return out_points, out_colors


if __name__ == '__main__':
    left_image = cv2.imread(r"F:\Deep Learning\NASA\images\left_images\1739066832.505062.png")
    right_image = cv2.imread(r"F:\Deep Learning\NASA\images\right_images\1739066832.505062.png")
    pcd = generate_pcd()
    pcd.get_pcd(left_image, right_image)

    
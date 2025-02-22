import os
import time

import numpy as np
import open3d as o3d
from scipy import stats

class PointCloudProcessor:
    def __init__(self, crop_radius=10.0, voxel_size=0.2, grid_resolution=0.1,
                 global_map_origin=(-50, -50), global_map_size=(100, 100)):
        self.crop_radius = crop_radius
        self.voxel_size = voxel_size
        self.grid_resolution = grid_resolution
        self.global_map_origin = np.array(global_map_origin)
        self.global_map_size = np.array(global_map_size)
        
        self.global_map_width = int(np.ceil(self.global_map_size[0] / self.grid_resolution))
        self.global_map_height = int(np.ceil(self.global_map_size[1] / self.grid_resolution))
        self.global_map = np.full((self.global_map_height, self.global_map_width), np.nan)

    def crop_pointcloud(self, pointcloud, center):
        cx, cy = center
        bounds = np.array([
            [cx - self.crop_radius, cx + self.crop_radius],
            [cy - self.crop_radius, cy + self.crop_radius]
        ])
        mask = (
            (pointcloud[:, 0] >= bounds[0, 0]) & 
            (pointcloud[:, 0] <= bounds[0, 1]) &
            (pointcloud[:, 1] >= bounds[1, 0]) & 
            (pointcloud[:, 1] <= bounds[1, 1])
        )
        return pointcloud[mask]

    def generate_2_5D_map(self, pointcloud, center):
        grid_size = int(np.ceil((2 * self.crop_radius) / self.grid_resolution))
        x_min, y_min = center[0] - self.crop_radius, center[1] - self.crop_radius
        x_max, y_max = x_min + 2*self.crop_radius, y_min + 2*self.crop_radius

        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        y_edges = np.linspace(y_min, y_max, grid_size + 1)

        if pointcloud.size == 0:
            return np.full((grid_size, grid_size), np.nan)

        x, y, z = pointcloud.T
        height_map, _, _, _ = stats.binned_statistic_2d(
            x, y, z,
            statistic='max',
            bins=[x_edges, y_edges]
        )
        return np.where(np.isnan(height_map.T), np.nan, height_map.T)

    def add_to_global_map(self, local_map, center):
        grid_size = local_map.shape[0]
        local_origin = np.array([center[0] - self.crop_radius, center[1] - self.crop_radius])
        
        i, j = np.indices((grid_size, grid_size))
        x_global = local_origin[0] + i * self.grid_resolution
        y_global = local_origin[1] + j * self.grid_resolution

        global_i = ((x_global - self.global_map_origin[0]) / self.grid_resolution).astype(int)
        global_j = ((y_global - self.global_map_origin[1]) / self.grid_resolution).astype(int)

        valid_mask = (
            (global_i >= 0) & (global_i < self.global_map_width) &
            (global_j >= 0) & (global_j < self.global_map_height)
        )
        valid_mask &= ~np.isnan(local_map)

        valid_global_i = global_i[valid_mask]
        valid_global_j = global_j[valid_mask]
        local_values = local_map[valid_mask]

        if valid_global_i.size == 0:
            return

        current_values = self.global_map[valid_global_j, valid_global_i]
        update_mask = np.isnan(current_values) | (local_values > current_values)

        self.global_map[valid_global_j[update_mask], valid_global_i[update_mask]] = local_values[update_mask]

    def process(self, pointcloud, center):
        # Single Open3D conversion pipeline
        cropped_np = self.crop_pointcloud(pointcloud, center)
        if cropped_np.size == 0:
            return np.array([])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cropped_np)
        
        # Combined Open3D operations
        _, clean_ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(clean_ind)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        downsampled_np = np.asarray(pcd.points) if pcd.has_points() else np.zeros((0, 3))
        local_map = self.generate_2_5D_map(downsampled_np, center)
        self.add_to_global_map(local_map, center)
        return local_map
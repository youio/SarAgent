import os
import time
import logging
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)

class PointCloudProcessor:
    def __init__(self, crop_radius=10.0, voxel_size=0.5, grid_resolution=0.5,
                 global_map_origin=(0, 0), global_map_size=(15, 15)):
        # Parameter validation
        if crop_radius <= 0:
            raise ValueError("crop_radius must be positive.")
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive.")
        if grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive.")
        if np.any(np.array(global_map_size) <= 0):
            raise ValueError("global_map_size must have positive dimensions.")
        if len(global_map_origin) != 2 or len(global_map_size) != 2:
            raise ValueError("global_map_origin and global_map_size must be 2-element tuples.")
        
        self.crop_radius = crop_radius
        self.voxel_size = voxel_size
        self.grid_resolution = grid_resolution
        self.global_map_origin = np.array(global_map_origin)
        self.global_map_size = np.array(global_map_size)
        
        self.global_map_width = int(np.ceil(self.global_map_size[0] / self.grid_resolution))
        self.global_map_height = int(np.ceil(self.global_map_size[1] / self.grid_resolution))
        self.global_map = np.full((self.global_map_height, self.global_map_width), np.nan)
        logging.info("PointCloudProcessor initialized successfully.")

    def crop_pointcloud(self, pointcloud, center):
        if not isinstance(pointcloud, np.ndarray) or pointcloud.shape[1] != 3:
            raise ValueError("pointcloud must be a numpy array of shape (N, 3).")
        if len(center) != 2:
            raise ValueError("Center must be a 2-element coordinate.")
        
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
        if pointcloud.size == 0:
            logging.warning("Empty pointcloud passed to generate_2_5D_map.")
            grid_size = int(np.ceil((2 * self.crop_radius) / self.grid_resolution))
            return np.full((grid_size, grid_size), np.nan)
        
        grid_size = int(np.ceil((2 * self.crop_radius) / self.grid_resolution))
        x_min, y_min = center[0] - self.crop_radius, center[1] - self.crop_radius
        x_edges = np.linspace(x_min, x_min + 2*self.crop_radius, grid_size + 1)
        y_edges = np.linspace(y_min, y_min + 2*self.crop_radius, grid_size + 1)

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
        try:
            if not isinstance(pointcloud, np.ndarray) or pointcloud.shape[1] != 3:
                raise ValueError("Input pointcloud must be a numpy array of shape (N, 3)")
            
            cropped_np = self.crop_pointcloud(pointcloud, center)
            if cropped_np.size == 0:
                logging.warning("Cropped pointcloud is empty.")
                return np.array([])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cropped_np)
            
            # Outlier removal
            _, clean_ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(clean_ind)
            
            # Voxel downsampling
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            
            downsampled_np = np.asarray(pcd.points) if pcd.has_points() else np.zeros((0, 3))
            local_map = self.generate_2_5D_map(downsampled_np, center)
            self.add_to_global_map(local_map, center)
            return local_map
        
        except Exception as e:
            logging.error(f"Error in processing point cloud: {e}")
            return np.array([])

    def visualize_global_map(self):
        if np.all(np.isnan(self.global_map)):
            logging.warning("Global map is empty - nothing to visualize.")
            return
        
        plt.figure(figsize=(10, 8))
        map_data = np.nan_to_num(self.global_map, nan=np.nanmin(self.global_map) - 1 if np.any(~np.isnan(self.global_map)) else 0)
        
        plt.imshow(map_data, 
                   origin='lower', 
                   extent=(self.global_map_origin[0], 
                           self.global_map_origin[0] + self.global_map_size[0],
                           self.global_map_origin[1], 
                           self.global_map_origin[1] + self.global_map_size[1]),
                   cmap='terrain')
        plt.colorbar(label='Height (m)')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title('Global 2.5D Height Map')
        plt.grid(visible=True, linestyle='--', alpha=0.5)
        plt.show()

# Example usage
if __name__ == "__main__":
    try:
        # Load point cloud
        pcd = o3d.io.read_point_cloud("6.ply")
        if not pcd.has_points():
            raise ValueError("Loaded point cloud is empty.")
        pointcloud_np = np.asarray(pcd.points)
        
        # Initialize processor
        processor = PointCloudProcessor()
        
        # Process with base location at (0,0)
        local_map = processor.process(pointcloud_np, (0, 0))
        
        # Visualize results
        print("Local map sample values:\n", local_map[:5, :5])  # Print first 5x5 values
        processor.visualize_global_map()
        
    except FileNotFoundError:
        logging.error("Point cloud file not found.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")

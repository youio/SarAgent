import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'terrain_mapper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class TerrainMapper:
    """
    A class to convert point clouds into a 2.5D terrain map in real-time.
    The map is built by aggregating point clouds captured from a moving vehicle,
    transforming them to a global frame, and estimating ground height per (x, y) cell.
    """
    
    def __init__(self, cell_size, z_min, z_max, nbins, min_x, max_x, min_y, max_y, 
                 ground_percentile=10, voxel_size=None, outlier_nb_neighbors=40, 
                 outlier_std_ratio=2.0):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing TerrainMapper")
        
        self.cell_size = cell_size
        self.z_min = z_min
        self.z_max = z_max
        self.nbins = nbins
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.ground_percentile = ground_percentile
        self.voxel_size = voxel_size
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        
        # Compute grid dimensions
        self.nx = int(np.ceil((max_x - min_x) / cell_size))
        self.ny = int(np.ceil((max_y - min_y) / cell_size))
        self.bin_size = (z_max - z_min) / nbins
        
        # Initialize histogram to store z-value distributions
        self.histogram = np.zeros((self.ny, self.nx, nbins), dtype=np.int32)
        # Initialize red_cells to track cells with red points
        self.red_cells = np.zeros((self.ny, self.nx), dtype=bool)
        self.logger.info(f"Grid initialized: {self.nx}x{self.ny} cells, {nbins} z-bins")

    def update_map(self, pointcloud, pose, crop_x=(-25,25), crop_y=(-25,25), crop_z=None, angle_rad=np.pi/2):
        """
        Update the global terrain map with a point cloud, optionally cropped in x, y, z.

        Parameters:
        - pointcloud (o3d.geometry.PointCloud): Input point cloud in local frame.
        - pose (np.ndarray): 4x4 transformation matrix from local to global frame.
        - crop_x (tuple, optional): (min_x, max_x) bounds for cropping in x (meters).
        - crop_y (tuple, optional): (min_y, max_y) bounds for cropping in y (meters).
        - crop_z (tuple, optional): (min_z, max_z) bounds for cropping in z (meters).
        """
        self.logger.info(f"Updating map with point cloud of {len(pointcloud.points)} points")
        
        if self.voxel_size is not None:
            pointcloud = pointcloud.voxel_down_sample(self.voxel_size)
            self.logger.info(f"Downsampled to {len(pointcloud.points)} points with voxel_size={self.voxel_size}")
        
        if angle_rad is not None:
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad),  np.cos(angle_rad)]
            ])
            pointcloud.rotate(R, center=(0, 0, 0))
            self.logger.info(f"Point cloud rotated by {angle_rad} degrees around Z-axis")

        
        pointcloud, _ = pointcloud.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors, 
            std_ratio=self.outlier_std_ratio
        )
        self.logger.info(f"Outliers removed, remaining points: {len(pointcloud.points)}")
        
        points = np.asarray(pointcloud.points)
        colors = np.asarray(pointcloud.colors)  # Get colors
        if len(points) == 0:
            self.logger.warning("No points left after processing; skipping update")
            return
        
        # Transform points to global coordinates
        points_global = (pose @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :3]
        
        # Apply cropping if specified
        mask_crop = np.ones(len(points_global), dtype=bool)  # Start with all points included
        if crop_x is not None:
            mask_crop &= (points_global[:, 0] >= crop_x[0]) & (points_global[:, 0] <= crop_x[1])
        if crop_y is not None:
            mask_crop &= (points_global[:, 1] >= crop_y[0]) & (points_global[:, 1] <= crop_y[1])
        if crop_z is not None:
            mask_crop &= (points_global[:, 2] >= crop_z[0]) & (points_global[:, 2] <= crop_z[1])
        
        points_global = points_global[mask_crop]
        colors = colors[mask_crop]
        self.logger.info(f"Points after cropping: {len(points_global)}")
        
        if len(points_global) == 0:
            self.logger.warning("No points remain after cropping; skipping update")
            return
        
        # Compute grid indices
        i = np.floor((points_global[:, 1] - self.min_y) / self.cell_size).astype(int)
        j = np.floor((points_global[:, 0] - self.min_x) / self.cell_size).astype(int)
        
        mask = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)
        i = i[mask]
        j = j[mask]
        z = points_global[mask, 2]
        colors = colors[mask]
        self.logger.debug(f"Points in grid bounds: {len(i)}")
        
        # Identify red points (RGB close to [1,0,0])
        is_red = (colors[:, 0] > 0.99) & (colors[:, 1] < 0.01) & (colors[:, 2] < 0.01)
        
        # Set z to 0 for red points
        z_modified = np.where(is_red, 0, z)
        
        k = np.floor((z_modified - self.z_min) / self.bin_size).astype(int)
        
        mask_bins = (k >= 0) & (k < self.nbins)
        i = i[mask_bins]
        j = j[mask_bins]
        k = k[mask_bins]
        self.logger.debug(f"Points in z-bin range: {len(k)}")
        
        np.add.at(self.histogram, (i, j, k), 1)
        
        # Mark cells with red points
        i_red = i[is_red[mask_bins]]
        j_red = j[is_red[mask_bins]]
        if len(i_red) > 0:
            self.red_cells[i_red, j_red] = True
        self.logger.info("Histogram and red cells updated successfully")

    def get_height_map(self):
        self.logger.info("Computing height map")
        total_count = np.sum(self.histogram, axis=2)
        cumsum = np.cumsum(self.histogram, axis=2).astype(float)
        cumsum /= total_count[:, :, np.newaxis] + 1e-6
        bin_idx = np.argmax(cumsum >= self.ground_percentile / 100, axis=2)
        z_ground = self.z_min + (bin_idx + 0.5) * self.bin_size
        
        # Initialize height map with nan
        height_map = np.full((self.ny, self.nx), np.nan)
        mask = total_count > 0
        height_map[mask] = z_ground[mask]
        # Set height to 0 for cells with red points
        height_map[self.red_cells] = 0
        self.logger.info("Height map computed")
        return height_map

    def get_visualization_pointcloud(self):
        self.logger.info("Generating visualization point cloud")
        height_map = self.get_height_map()
        x = np.linspace(self.min_x + self.cell_size / 2, 
                       self.max_x - self.cell_size / 2, self.nx)
        y = np.linspace(self.min_y + self.cell_size / 2, 
                       self.max_y - self.cell_size / 2, self.ny)
        xx, yy = np.meshgrid(x, y)
        points = np.stack((xx.ravel(), yy.ravel(), height_map.ravel()), axis=1)
        valid = ~np.isnan(points[:, 2])
        points = points[valid]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if len(points) > 0:
            z_min = np.min(points[:, 2])
            z_max = np.max(points[:, 2])
            if z_max > z_min:
                colors = plt.get_cmap('viridis')((points[:, 2] - z_min) / (z_max - z_min))[:, :3]
                pcd.colors = o3d.utility.Vector3dVector(colors)
            self.logger.info(f"Visualization point cloud generated with {len(points)} points")
        else:
            self.logger.warning("No valid points for visualization")
        
        return pcd

    def load_and_process_ply(self, ply_file, pose=None):
        """
        Load a point cloud from a .ply file, optionally rotate it, update the map, and return it.
        
        Parameters:
        - ply_file (str): Path to the .ply file.
        - pose (np.ndarray, optional): 4x4 transformation matrix; defaults to identity if None.
        - rotation_angle_deg (float, optional): Rotation angle in degrees around Z-axis; defaults to 0.
        
        Returns:
        - pointcloud (o3d.geometry.PointCloud): Loaded and rotated point cloud.
        """
        self.logger.info(f"Loading point cloud from {ply_file}")
        if pose is None:
            pose = np.eye(4)

        
        pointcloud = o3d.io.read_point_cloud(ply_file)
        if not pointcloud.has_points():
            self.logger.error(f"Failed to load point cloud from {ply_file} or it is empty")
            raise ValueError(f"Failed to load point cloud from {ply_file} or it is empty")
        
        self.logger.info(f"Loaded {len(pointcloud.points)} points from {ply_file}")
        
        self.update_map(pointcloud, pose)
        return pointcloud

    def visualize_map(self, pointcloud=None):
        """
        Visualize the current 2.5D terrain map using Matplotlib as a 2D heatmap.
        Cells with red points are colored red.
        
        Parameters:
        - pointcloud (o3d.geometry.PointCloud, optional): Original point cloud (not used in this version).
        """
        self.logger.info("Starting map visualization with Matplotlib")
        
        # Get the height map
        height_map = self.get_height_map()
        
        if np.all(np.isnan(height_map)):
            self.logger.warning("Height map contains no valid data to visualize")
            return
        
        # Create x and y coordinate grids for plotting
        x = np.linspace(self.min_x, self.max_x, self.nx)
        y = np.linspace(self.min_y, self.max_y, self.ny)
        X, Y = np.meshgrid(x, y)
        
        # Normalize height for colormap
        vmin = np.nanmin(height_map)
        vmax = np.nanmax(height_map)
        norm_height = (height_map - vmin) / (vmax - vmin)
        
        # Create RGBA colors using viridis
        colors = plt.get_cmap('viridis')(norm_height)
        
        # Set alpha to 0 for nan values (transparent)
        colors[np.isnan(norm_height), 3] = 0.0
        
        # Set red for cells with red points
        colors[self.red_cells, 0] = 1.0
        colors[self.red_cells, 1] = 0.0
        colors[self.red_cells, 2] = 0.0
        colors[self.red_cells, 3] = 1.0
        
        # Plot the RGB image
        plt.figure(figsize=(10, 8))
        plt.imshow(colors, extent=[self.min_x, self.max_x, self.min_y, self.max_y], origin='lower')
        plt.title('2.5D Terrain Map (Red cells indicate special areas)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        # Add a colorbar for height reference
        # Note: The colorbar will not include the red override
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Height (m)')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
        
        self.logger.info("Displaying Matplotlib plot")
        plt.show()
        self.logger.info("Matplotlib visualization completed")

# Usage Example
if __name__ == "__main__":
    mapper = TerrainMapper(
        cell_size=0.25, z_min=-100, z_max=100, nbins=100,
        min_x=-100, max_x=100, min_y=-100, max_y=100,
        voxel_size=0.1, ground_percentile=75
    )

    ply_file = "6.ply"  # Replace with actual path
    
    try:
        original_pcd = mapper.load_and_process_ply(ply_file)
        mapper.visualize_map(pointcloud=original_pcd)
    except Exception as e:
        logging.error(f"Main execution failed: {e}")


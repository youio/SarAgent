# import open3d as o3d

# # Load the .ply file
# pcd = o3d.io.read_point_cloud(r"F:\Deep Learning\NASA\output\ply\4.ply")

# # Visualize
# o3d.visualization.draw_geometries([pcd])
# exit(0)


import open3d as o3d
import numpy as np

# Load the .ply file
pcd = o3d.io.read_point_cloud(r"F:\Deep Learning\NASA\output\ply\5.ply")

# Convert point cloud to numpy array
points = np.asarray(pcd.points)

# Define depth range
depth_min = -20
depth_max = -5

# Filter points based on the z-coordinate (depth)
filtered_points = points[(points[:, 2] >= depth_min) & (points[:, 2] <= depth_max)]

# Create a new point cloud with the filtered points
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

# Copy colors from the original point cloud if present
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    filtered_colors = colors[(points[:, 2] >= depth_min) & (points[:, 2] <= depth_max)]
    filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

# Visualize the filtered point cloud
o3d.visualization.draw_geometries([filtered_pcd])


exit(0)







import open3d as o3d
import numpy as np

# Load the .ply file
pcd = o3d.io.read_point_cloud(r"F:\Deep Learning\NASA\output\ply\3.ply")

# Use VisualizerWithEditing to select points
print("Use [Shift + Left Click] to select points, and [Ctrl + Left Click] to deselect points.")
print("Press [Q] or close the window to finish the selection.")

# Visualize and select points
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # User selects points in the window
vis.destroy_window()

# Retrieve the indices of the selected points
selected_indices = vis.get_picked_points()

# Print the selected points' indices and coordinates
if selected_indices:
    points = np.asarray(pcd.points)
    for idx in selected_indices:
        print(f"Point Index: {idx}, Coordinates: {points[idx]}")
else:
    print("No points were selected.")

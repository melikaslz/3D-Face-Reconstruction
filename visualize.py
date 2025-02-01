import open3d as o3d
import pandas as pd
import numpy as np

# Load keypoints with depth
landmarks_df = pd.read_csv("facial_keypoints.csv")

# Extract first frame keypoints
frame_keypoints = landmarks_df[landmarks_df["Frame"] ==100][["X", "Y", "Z"]].values

# Convert to Open3D point cloud format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(frame_keypoints)

# Visualize 3D keypoints
o3d.visualization.draw_geometries([pcd])

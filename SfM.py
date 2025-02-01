import numpy as np
import pandas as pd
import cv2
from scipy.optimize import least_squares

# Load the keypoints CSV file
def load_keypoints(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Load the data
csv_path = "facial_keypoints.csv"
df = load_keypoints(csv_path)


def estimate_fundamental_matrix(pts1, pts2):
    """Estimate Fundamental Matrix using 8-point algorithm"""
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)
    return F, mask

# Select two consecutive frames for SfM
frame1 = df[df["Frame"] == 0][["X", "Y"]].values
frame2 = df[df["Frame"] == 1][["X", "Y"]].values

# Compute Fundamental Matrix
F, mask = estimate_fundamental_matrix(frame1, frame2)
print("Fundamental Matrix:\n", F)

K = np.array([[480, 0, 240],
              [0, 480, 426],
              [0, 0, 1]])

E = K.T @ F @ K  # Essential Matrix
print("Essential Matrix:\n", E)

# Decompose Essential Matrix
U, _, Vt = np.linalg.svd(E)
W = np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 1]])

# Two possible rotation matrices
R1 = U @ W @ Vt
R2 = U @ W.T @ Vt
T = U[:, 2]  # Translation (up to scale)

print("Possible Rotation Matrices:\n", R1, "\n\n", R2)
print("Translation Vector:\n", T)


def triangulate_points(K, R, T, pts1, pts2):
    """Triangulate points given camera motion"""
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for frame 1
    P2 = K @ np.hstack((R, T.reshape(3, 1)))  # Projection matrix for frame 2

    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T
    pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3).T

    X_h = cv2.triangulatePoints(P1, P2, pts1_h[:2], pts2_h[:2])
    X = X_h[:3] / X_h[3]  # Convert from homogeneous to 3D

    return X.T

# Compute 3D points
X_3D = triangulate_points(K, R1, T, frame1, frame2)

# Update depth (z) in dataframe
df.loc[df["Frame"] == 0, "Z"] = X_3D[:, 2]

#Save updated keypoints
df.to_csv("updated_depth_keypoints.csv", index=False, header=False)
print("Updated depth values saved to 'updated_depth_keypoints.csv'")


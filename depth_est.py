import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
from torchvision import models
from PIL import Image

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # Lightweight version
midas.eval()

# Define image transformation for MiDaS
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize for MiDaS
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # MiDaS normalization
])

# Load detected facial keypoints
landmarks_df = pd.read_csv("facial_keypoints.csv")

# Open the video file
video_path = "face.mp4"
cap = cv2.VideoCapture(video_path)

# Store updated keypoints with depth
updated_landmarks = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract keypoints for the current frame
    frame_keypoints = landmarks_df[landmarks_df["Frame"] == frame_idx]

    if not frame_keypoints.empty:
        # Convert frame to PIL Image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Transform image for MiDaS
        input_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            depth_map = midas(input_tensor)  # Predict depth map

        depth_map = depth_map.squeeze().numpy()  # Convert to NumPy array
        depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))  # Resize depth map

        # Get the depth for each keypoint
        for _, row in frame_keypoints.iterrows():
            x, y = int(row["X"]), int(row["Y"])

            # Ensure x, y are within bounds
            x = np.clip(x, 0, depth_map_resized.shape[1] - 1)
            y = np.clip(y, 0, depth_map_resized.shape[0] - 1)

            z_depth = depth_map_resized[y, x]  # Get depth value

            # Store updated keypoint with estimated depth
            updated_landmarks.append([row["Frame"], row["Landmark_ID"], row["X"], row["Y"], z_depth])

    frame_idx += 1

# Release video
cap.release()

# Convert updated keypoints to DataFrame
columns = ["Frame", "Landmark_ID", "X", "Y", "Z"]
updated_landmarks_df = pd.DataFrame(updated_landmarks, columns=columns)



# Save to CSV
updated_landmarks_df.to_csv("facial_keypoints_with_depth.csv", index=False)

print("Updated keypoints with depth saved to facial_keypoints_with_depth.csv")

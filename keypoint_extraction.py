import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

print(mp_face_mesh)
print("_____________")
print(face_mesh)
# Open video file
video_path = "face.mp4"  # Change to your video path
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video Info: {fps} FPS, {frame_count} frames, {width}x{height}")

# Store results
landmarks_list = []

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB (MediaPipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with FaceMesh
    results = face_mesh.process(rgb_frame)

    # Extract keypoints if a face is detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            keypoints = []
            for i, landmark in enumerate(face_landmarks.landmark):
                x, y, z = landmark.x * width, landmark.y * height, landmark.z  # Scale x, y to pixel coordinates
                keypoints.append([frame_idx, i, x, y, z])  # Store frame index, landmark ID, and coordinates
            
            landmarks_list.extend(keypoints)

    frame_idx += 1

# Release video
cap.release()

# Convert to DataFrame
columns = ["Frame", "Landmark_ID", "X", "Y", "Z"]
landmarks_df = pd.DataFrame(landmarks_list, columns=columns)

# Save to CSV
landmarks_df.to_csv("facial_keypoints.csv", index=False)

print("Facial keypoints saved to facial_keypoints.csv")

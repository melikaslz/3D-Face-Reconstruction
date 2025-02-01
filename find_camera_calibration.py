import cv2
import numpy as np

# --- Parameters ---
video_path = "face.mp4"  # Replace with your video file
orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- Read Two Frames from the Video ---
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()
if not ret:
    raise ValueError("Cannot read first frame from video.")

ret, frame2 = cap.read()
if not ret:
    raise ValueError("Cannot read second frame from video.")
cap.release()

# Convert frames to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# --- Feature Detection and Matching ---
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

if des1 is None or des2 is None:
    raise ValueError("No descriptors found. Make sure your video has enough texture.")

# Match descriptors between frames
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Use only the best matches (you may tune this number)
num_matches = min(100, len(matches))
matches = matches[:num_matches]

# Extract matched points
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# --- Initial Intrinsics Guess ---
# Here we assume the focal length is approximately the image width (in pixels)
# and the principal point is at the center.
img_h, img_w = gray1.shape
focal_length = img_w  # this is a rough guess; you might refine this
K = np.array([[focal_length, 0, img_w / 2],
              [0, focal_length, img_h / 2],
              [0, 0, 1]], dtype=float)

# --- Compute Essential Matrix and Recover Pose ---
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
if E is None:
    raise ValueError("Essential matrix estimation failed.")

# Recover relative pose (rotation R and translation t) from the essential matrix.
_, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

print("Estimated rotation between frame1 and frame2:")
print(R)
print("\nEstimated translation (up to scale) between frame1 and frame2:")
print(t)
print(K)
import cv2

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

video_path = "face.mp4"
resolution = get_video_resolution(video_path)
print(f"Video resolution: {resolution[0]}x{resolution[1]}")

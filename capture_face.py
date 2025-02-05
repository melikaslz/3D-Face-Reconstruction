import cv2

# Capture live video feed
cap = cv2.VideoCapture(0)  # 0 for the default webcam


while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

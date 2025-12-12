import cv2
from ultralytics import YOLO
import sys

# Load the YOLO model
model = YOLO(sys.argv[1])  # Replace with your model path if needed

# Open the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model.predict(frame)
    annotated_frame = results[0].plot()  # YOLO → image with detections

    # Display the annotated frame
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

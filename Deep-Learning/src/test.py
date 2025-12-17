import cv2
import sys
import os

from ultralytics import YOLO

video_filepath = sys.argv[1]
model_pt_filepath = sys.argv[2]

model = YOLO(model_pt_filepath)

cap = cv2.VideoCapture(video_filepath)

# Read video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Choose codec (works for .mov or .mp4)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Output video writer
out = cv2.VideoWriter(f"{os.path.basename(video_filepath)}_results.mov", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    annotated_frame = results[0].plot()  # YOLO → image with detections

    out.write(annotated_frame)  # Save frame to video

cap.release()
out.release()
cv2.destroyAllWindows()




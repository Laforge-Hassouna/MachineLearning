from ultralytics import YOLO

model = YOLO()

# Train the model
results = model.train(data="datasets/rubber_duck.yaml", epochs=10, imgsz=640)
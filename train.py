#importing libraries
from ultralytics import YOLO
#defining a model
model = YOLO("yolov8n.pt")

model.train(
    data="database/dataset.yaml",
    epochs=10,
    imgsz=640,
    batch=16
)
from ultralytics import YOLO

# Load the nano model
model = YOLO("yolo11n.pt")

# Export to NCNN format for high-speed CPU inference
model.export(format="ncnn")
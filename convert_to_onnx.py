from ultralytics import YOLO

model = YOLO('./runs/detect/license_plate_detector/weights/best.pt')
model.export(
    format="onnx",
    opset=12,
    simplify=True,
    imgsz=320
)
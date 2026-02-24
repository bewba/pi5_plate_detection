from ultralytics import YOLO

model = YOLO("yolo11n.pt")

print('training model!')

results = model.train(
    data="License-Plate-1/data.yaml",  # path to your yaml file
    epochs=100,                        # number of training epochs
    imgsz=640,                         # image size
    device=0,                          # use '0' for GPU, or 'cpu' if no GPU
    name="license_plate_detector"      # name of the project folder
)

print('done training model!')

model.export(format="ncnn")
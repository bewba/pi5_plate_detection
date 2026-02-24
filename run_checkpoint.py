from ultralytics import YOLO

print('resuming training!')

# Load the interrupted checkpoint
model = YOLO('./runs/detect/license_plate_detector/weights/last.pt')

print('model taken!')

# Resume training
model.train(resume=True)

print('done training!')

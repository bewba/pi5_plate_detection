import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

# -------------------------
# 1. Setup
# -------------------------
model = YOLO('./runs/detect/license_plate_detector/weights/best.pt')
model.to('cpu')
reader = easyocr.Reader(['en'], gpu=False)  # Use CPU to avoid ROCm GPU issue

# Create run folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs/detect", f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

output_video_path = os.path.join(run_dir, "output.mp4")
plate_save_dir = os.path.join(run_dir, "plates")
os.makedirs(plate_save_dir, exist_ok=True)

# -------------------------
# 2. OCR preprocessing (only when clear plate is ready)
# -------------------------
def preprocess_for_ocr(plate_img):
    plate_img = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# -------------------------
# 3. Helper functions
# -------------------------
def iou(boxA, boxB):
    # Intersection over union for two boxes (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# -------------------------
# 4. Video Setup
# -------------------------
video_path = 'demo.mp4'
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

side_panel_width = 300
out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width + side_panel_width, height)  # combined width
)

print("Processing video with 3–5 frame plate confirmation...")

# -------------------------
# 5. Tracking state
# -------------------------
active_plates = {}  # plate_id -> {'box':(x1,y1,x2,y2), 'frames': count, 'ocr_done': bool}
next_plate_id = 0
recognized_plates = set()
plate_counter = 0

# -------------------------
# 6. Detection loop
# -------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    side_panel_width = 300
    side_panel = np.zeros((frame.shape[0], side_panel_width, 3), dtype=np.uint8)

    if side_panel.shape[0] != frame.shape[0]:
        pad_height = frame.shape[0] - side_panel.shape[0]
        side_panel = cv2.copyMakeBorder(side_panel, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))


    results = model(frame, conf=0.7, verbose=False)
    current_boxes = []

    # Collect current frame detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < 120 or h < 40:  # ignore tiny plates
                continue
            current_boxes.append((x1, y1, x2, y2))

    # Update active plates
    matched_ids = set()
    for box in current_boxes:
        matched = False
        for pid, data in active_plates.items():
            if iou(box, data['box']) > 0.3:
                # Update box and increment frame count
                data['box'] = box
                data['frames'] += 1
                matched_ids.add(pid)
                matched = True
                break
        if not matched:
            # New plate
            active_plates[next_plate_id] = {'box': box, 'frames': 1, 'ocr_done': False}
            matched_ids.add(next_plate_id)
            next_plate_id += 1

    # Remove plates not detected this frame
    to_delete = [pid for pid in active_plates if pid not in matched_ids]
    for pid in to_delete:
        del active_plates[pid]

    # Create side panel for plates
    side_panel = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)  # 300px wide panel

    # OCR for plates seen in 3–5 consecutive frames
    panel_y = 10  # vertical start in side panel
    for pid, data in active_plates.items():
        x1, y1, x2, y2 = data['box']
        if data['frames'] >= 3 and not data['ocr_done']:
            plate_crop = frame[y1:y2, x1:x2]
            raw_plate_path = os.path.join(plate_save_dir, f"plate_{plate_counter}.jpg")
            cv2.imwrite(raw_plate_path, plate_crop)

            processed = preprocess_for_ocr(plate_crop)
            ocr_results = reader.readtext(
                processed,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
                paragraph=False
            )

            # Find highest confidence result
            best_text, best_prob = None, 0
            for _, text, prob in ocr_results:
                if prob > best_prob:
                    best_prob = prob
                    best_text = text.strip()

            # Only display if confidence >= 85%
            if best_text and best_prob >= 0.85:
                recognized_plates.add(best_text)
                display_label = f"{best_text} ({int(best_prob*100)}%)"

                # Draw on main frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, display_label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Resize plate_crop to fit side panel width
                h_crop, w_crop = plate_crop.shape[:2]
                scale = 280 / max(w_crop, h_crop)
                plate_resized = cv2.resize(plate_crop, (int(w_crop*scale), int(h_crop*scale)))
                
                # Paste plate crop onto side panel
                y_end = panel_y + plate_resized.shape[0]
                side_panel[panel_y:y_end, 10:10+plate_resized.shape[1]] = plate_resized
                cv2.putText(side_panel, display_label, (10, y_end + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                panel_y = y_end + 50  # next plate

                data['ocr_done'] = True
                plate_counter += 1

    # Concatenate main frame and side panel
    combined_frame = np.hstack([frame, side_panel])
    out.write(combined_frame)

cap.release()
out.release()
print(f"Done! Results saved to {run_dir}")
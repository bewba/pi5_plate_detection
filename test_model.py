import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from collections import Counter
import time  # <-- add this at the top

start_time = time.time()

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
# 2. OCR preprocessing
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
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def run_ocr(plate_img):
    """Run OCR on a plate image and return (text, confidence) or (None, 0)."""
    processed = preprocess_for_ocr(plate_img)
    ocr_results = reader.readtext(
        processed,
        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
        paragraph=False
    )
    best_text, best_prob = None, 0
    for _, text, prob in ocr_results:
        if prob > best_prob:
            best_prob = prob
            best_text = text.strip()
    return best_text, best_prob

def draw_sidebar(panel, confirmed_plates):
    """
    Draw confirmed plates onto the sidebar panel.
    confirmed_plates: list of dicts with 'plate_img', 'text', 'conf'
    """
    panel[:] = 30  # dark background
    panel_w = panel.shape[1]

    # Title
    cv2.putText(panel, "DETECTED PLATES", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.line(panel, (10, 38), (panel_w - 10, 38), (80, 80, 80), 1)

    y = 50
    for entry in confirmed_plates:
        plate_img = entry['plate_img']
        label = f"{entry['text']}  {entry['conf']}%"

        # Resize plate thumbnail to fit panel
        h_crop, w_crop = plate_img.shape[:2]
        scale = (panel_w - 20) / max(w_crop, 1)
        new_w = int(w_crop * scale)
        new_h = int(h_crop * scale)
        new_h = min(new_h, 70)  # cap height
        scale2 = new_h / max(h_crop, 1)
        new_w2 = int(w_crop * scale2)
        thumb = cv2.resize(plate_img, (new_w2, new_h))

        y_end = y + new_h
        if y_end > panel.shape[0] - 30:
            break  # no more space

        panel[y:y_end, 10:10 + new_w2] = thumb

        # Label below the thumbnail
        cv2.putText(panel, label, (10, y_end + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
        y = y_end + 40

        # Divider
        if y < panel.shape[0] - 10:
            cv2.line(panel, (10, y - 5), (panel_w - 10, y - 5), (60, 60, 60), 1)

# -------------------------
# 4. Video Setup
# -------------------------
video_path = 'demo.mp4'
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

SIDE_W = 320
out = cv2.VideoWriter(output_video_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (width + SIDE_W, height))

print("Processing video — requires 5 consistent OCR frames at 95% agreement...")

# -------------------------
# 5. Tracking state
# -------------------------
# active_plates: plate_id -> {
#   'box': (x1,y1,x2,y2),
#   'ocr_done': bool,
#   'ocr_buffer': [(text, conf), ...],   # up to 5 OCR results
#   'best_frame': frame_img or None,     # highest-conf frame crop seen so far
#   'best_conf': float
# }
active_plates = {}
next_plate_id = 0
plate_counter = 0

# Persistent sidebar entries: list of {'plate_img', 'text', 'conf'}
confirmed_plates = []
recognized_texts = set()

# Pre-render the sidebar once; only re-render when a new plate is confirmed
sidebar_panel = np.full((height, SIDE_W, 3), 30, dtype=np.uint8)
cv2.putText(sidebar_panel, "DETECTED PLATES", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
cv2.line(sidebar_panel, (10, 38), (SIDE_W - 10, 38), (80, 80, 80), 1)

# -------------------------
# 6. Detection loop
# -------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.7, verbose=False)
    current_boxes = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < 120 or h < 40:
                continue
            current_boxes.append((x1, y1, x2, y2))

    # ------- Match detections to tracked plates -------
    matched_ids = set()
    for box in current_boxes:
        matched = False
        for pid, data in active_plates.items():
            if iou(box, data['box']) > 0.3:
                data['box'] = box
                matched_ids.add(pid)
                matched = True
                break
        if not matched:
            active_plates[next_plate_id] = {
                'box': box,
                'ocr_done': False,
                'ocr_buffer': [],
                'best_frame': None,
                'best_conf': 0.0,
            }
            matched_ids.add(next_plate_id)
            next_plate_id += 1

    # Remove lost tracks
    for pid in [p for p in active_plates if p not in matched_ids]:
        del active_plates[pid]

    # ------- OCR accumulation: collect up to 5 frames per plate -------
    for pid, data in active_plates.items():
        if data['ocr_done']:
            # Still draw box + label from confirmed data
            x1, y1, x2, y2 = data['box']
            label = data.get('confirmed_label', '')
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if label:
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            continue

        x1, y1, x2, y2 = data['box']
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue

        text, conf = run_ocr(plate_crop)

        if text and conf >= 0.5:  # low bar here; real filter is the 5-frame consensus
            data['ocr_buffer'].append((text, conf))
            if conf > data['best_conf']:
                data['best_conf'] = conf
                data['best_frame'] = plate_crop.copy()

        # Draw tentative box while accumulating
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        pending = len(data['ocr_buffer'])
        cv2.putText(frame, f"Reading... {pending}/5", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # ------- Check 5-frame consensus -------
        if len(data['ocr_buffer']) >= 5:
            texts = [t for t, _ in data['ocr_buffer'][-5:]]
            counter = Counter(texts)
            most_common_text, count = counter.most_common(1)[0]
            agreement = count / 5  # ratio out of 5 frames

            if agreement >= 0.95 and most_common_text not in recognized_texts:
                # Get average confidence for the winning text
                winning_confs = [c for t, c in data['ocr_buffer'][-5:] if t == most_common_text]
                avg_conf = sum(winning_confs) / len(winning_confs)
                conf_pct = int(avg_conf * 100)

                recognized_texts.add(most_common_text)
                display_label = f"{most_common_text} ({conf_pct}%)"
                data['ocr_done'] = True
                data['confirmed_label'] = display_label

                # ------- Save images -------
                best_crop = data['best_frame'] if data['best_frame'] is not None else plate_crop

                # Save plate crop
                plate_path = os.path.join(plate_save_dir,
                                          f"plate_{plate_counter}_{most_common_text}.jpg")
                cv2.imwrite(plate_path, best_crop)

                # Save full frame with bounding box drawn
                frame_annotated = frame.copy()
                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_annotated, display_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                full_frame_path = os.path.join(plate_save_dir,
                                               f"frame_{plate_counter}_{most_common_text}.jpg")
                cv2.imwrite(full_frame_path, frame_annotated)

                plate_counter += 1
                print(f"  ✓ Confirmed plate: {most_common_text} ({conf_pct}%) "
                      f"[{int(agreement*100)}% frame agreement]")

                # ------- Update persistent sidebar -------
                confirmed_plates.append({
                    'plate_img': best_crop.copy(),
                    'text': most_common_text,
                    'conf': conf_pct,
                })
                sidebar_panel = np.full((height, SIDE_W, 3), 30, dtype=np.uint8)
                draw_sidebar(sidebar_panel, confirmed_plates)

            elif agreement < 0.95 and len(data['ocr_buffer']) >= 5:
                # Not enough agreement; slide the window forward (keep last 4)
                data['ocr_buffer'] = data['ocr_buffer'][-4:]

            # Update box colour once confirmed
            if data['ocr_done']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, data['confirmed_label'], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ------- Compose output frame -------
    combined_frame = np.hstack([frame, sidebar_panel])
    out.write(combined_frame)

cap.release()
out.release()
print(f"\nDone! Results saved to: {run_dir}")
print(f"  • Annotated video : {output_video_path}")
print(f"  • Plate crops/frames: {plate_save_dir}")


end_time = time.time()
elapsed = end_time - start_time
print(f"Processing time: {elapsed:.2f} seconds")
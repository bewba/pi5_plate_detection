import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# -------------------------
# 1. Setup
# -------------------------
model = YOLO('./runs/detect/license_plate_detector/weights/best.onnx')
model.to('cpu')

# Single shared reader — safe to call from multiple threads once created.
# Use a lock since EasyOCR's internal state isn't guaranteed thread-safe.
reader = easyocr.Reader(['en'], gpu=False)
ocr_lock = threading.Lock()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs/detect", f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

output_video_path = os.path.join(run_dir, "output.mp4")
plate_save_dir = os.path.join(run_dir, "plates")
os.makedirs(plate_save_dir, exist_ok=True)

# -------------------------
# 2. Tuning knobs  ← adjust these to trade speed vs accuracy
# -------------------------
DETECT_EVERY_N_FRAMES = 3     # Run YOLO every N frames; interpolate boxes in between
OCR_EVERY_N_FRAMES    = 2      # Submit an OCR job for each plate every N frames
OCR_WORKERS           = 2      # Parallel OCR threads (tune to your CPU core count)
REQUIRED_FRAMES       = 2      # Frames needed for consensus check
CONSENSUS_THRESHOLD   = 1   # 100% of frames must agree
MIN_OCR_CONF          = 0.70   # Minimum single-frame OCR confidence to count
YOLO_CONF             = 0.70
MIN_PLATE_W           = 120
MIN_PLATE_H           = 40

# -------------------------
# 3. OCR helpers
# -------------------------
def preprocess_for_ocr(plate_img):
    plate_img = cv2.resize(plate_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def run_ocr(plate_img):
    """Thread-safe OCR. Returns (text, conf) or (None, 0)."""
    processed = preprocess_for_ocr(plate_img)
    with ocr_lock:
        results = reader.readtext(
            processed,
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
            paragraph=False
        )
    best_text, best_prob = None, 0.0
    for _, text, prob in results:
        if prob > best_prob:
            best_prob = prob
            best_text = text.strip()
    return best_text, best_prob

# -------------------------
# 4. IoU
# -------------------------
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    aA = (a[2] - a[0]) * (a[3] - a[1])
    aB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(aA + aB - inter + 1e-6)

# -------------------------
# 5. Sidebar renderer
# -------------------------
def draw_sidebar(height, panel_w, confirmed_plates):
    panel = np.full((height, panel_w, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "DETECTED PLATES", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.line(panel, (10, 38), (panel_w - 10, 38), (80, 80, 80), 1)
    y = 50
    for entry in confirmed_plates:
        img = entry['plate_img']
        h_c, w_c = img.shape[:2]
        new_h = min(70, int(h_c * (panel_w - 20) / max(w_c, 1)))
        new_w = int(w_c * (new_h / max(h_c, 1)))
        thumb = cv2.resize(img, (new_w, new_h))
        y_end = y + new_h
        if y_end > height - 30:
            break
        panel[y:y_end, 10:10 + new_w] = thumb
        cv2.putText(panel, f"{entry['text']}  {entry['conf']}%",
                    (10, y_end + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
        y = y_end + 40
        if y < height - 10:
            cv2.line(panel, (10, y - 5), (panel_w - 10, y - 5), (60, 60, 60), 1)
    return panel

# -------------------------
# 6. Video setup
# -------------------------
video_path = 'demo.mp4'
cap   = cv2.VideoCapture(video_path)
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

SIDE_W = 320
out = cv2.VideoWriter(output_video_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (vid_w + SIDE_W, vid_h))

print(f"Processing {total} frames | YOLO every {DETECT_EVERY_N_FRAMES}f "
      f"| OCR every {OCR_EVERY_N_FRAMES}f | {OCR_WORKERS} OCR workers")

# -------------------------
# 7. State
# -------------------------
# active_plates: pid -> {box, ocr_done, ocr_buffer, best_frame, best_conf,
#                        confirmed_label, frames_since_ocr, pending_ocr}
active_plates    = {}
next_pid         = 0
plate_counter    = 0
recognized_texts = set()
confirmed_plates = []
sidebar_panel    = draw_sidebar(vid_h, SIDE_W, confirmed_plates)

executor         = ThreadPoolExecutor(max_workers=OCR_WORKERS)
ocr_result_queue = queue.Queue()

def ocr_job(pid, crop):
    """Worker: run OCR then push result onto the shared queue."""
    text, conf = run_ocr(crop)
    ocr_result_queue.put((pid, crop, text, conf))

# -------------------------
# 8. Main loop
# -------------------------
frame_idx  = 0
last_boxes = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLO (skipped frames reuse last result) ---
    if frame_idx % DETECT_EVERY_N_FRAMES == 0:
        yolo_out   = model(frame, conf=YOLO_CONF, verbose=False)
        last_boxes = []
        for result in yolo_out:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) >= MIN_PLATE_W and (y2 - y1) >= MIN_PLATE_H:
                    last_boxes.append((x1, y1, x2, y2))
    current_boxes = last_boxes

    # --- Track matching ---
    matched_ids = set()
    for box in current_boxes:
        matched = False
        for pid, data in active_plates.items():
            if iou(box, data['box']) > 0.3:
                data['box'] = box
                data['frames_since_ocr'] += 1
                matched_ids.add(pid)
                matched = True
                break
        if not matched:
            active_plates[next_pid] = {
                'box': box, 'ocr_done': False,
                'ocr_buffer': [], 'best_frame': None,
                'best_conf': 0.0, 'confirmed_label': '',
                'frames_since_ocr': 0, 'pending_ocr': False,
            }
            matched_ids.add(next_pid)
            next_pid += 1

    for pid in [p for p in active_plates if p not in matched_ids]:
        del active_plates[pid]

    # --- Drain completed OCR results ---
    while not ocr_result_queue.empty():
        try:
            pid, crop, text, conf = ocr_result_queue.get_nowait()
        except queue.Empty:
            break

        if pid not in active_plates:
            continue

        data = active_plates[pid]
        data['pending_ocr'] = False

        if text:
            print(f"    [OCR] pid={pid} '{text}' conf={conf:.2f} buf={len(data['ocr_buffer'])}")
        if text and conf >= MIN_OCR_CONF:
            data['ocr_buffer'].append((text, conf))
            if conf > data['best_conf']:
                data['best_conf']  = conf
                data['best_frame'] = crop.copy()

        # --- Consensus check ---
        if len(data['ocr_buffer']) >= REQUIRED_FRAMES:
            texts = [t for t, _ in data['ocr_buffer'][-REQUIRED_FRAMES:]]
            most_common, count = Counter(texts).most_common(1)[0]
            agreement = count / REQUIRED_FRAMES

            if agreement >= CONSENSUS_THRESHOLD and most_common not in recognized_texts:
                win_confs = [c for t, c in data['ocr_buffer'][-REQUIRED_FRAMES:]
                             if t == most_common]
                conf_pct  = int(sum(win_confs) / len(win_confs) * 100)
                label     = f"{most_common} ({conf_pct}%)"

                recognized_texts.add(most_common)
                data['ocr_done']        = True
                data['confirmed_label'] = label

                best_crop = data['best_frame'] if data['best_frame'] is not None else crop
                x1, y1, x2, y2 = data['box']

                # Save plate crop
                cv2.imwrite(os.path.join(plate_save_dir,
                    f"plate_{plate_counter}_{most_common}.jpg"), best_crop)

                # Save annotated full frame
                ann = frame.copy()
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(ann, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(plate_save_dir,
                    f"frame_{plate_counter}_{most_common}.jpg"), ann)

                plate_counter += 1
                print(f"  ✓ Frame {frame_idx:5d} | {most_common} {conf_pct}% "
                      f"[{int(agreement*100)}% agreement]")

                confirmed_plates.append({
                    'plate_img': best_crop.copy(),
                    'text': most_common, 'conf': conf_pct,
                })
                # Re-render sidebar only when a new plate is confirmed
                sidebar_panel = draw_sidebar(vid_h, SIDE_W, confirmed_plates)

            elif agreement < CONSENSUS_THRESHOLD:
                # Slide window — drop oldest result and try again next time
                data['ocr_buffer'] = data['ocr_buffer'][-(REQUIRED_FRAMES - 1):]

    # --- Submit new OCR jobs (non-blocking) ---
    for pid, data in active_plates.items():
        if data['ocr_done'] or data['pending_ocr']:
            continue
        if data['frames_since_ocr'] >= OCR_EVERY_N_FRAMES:
            x1, y1, x2, y2 = data['box']
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                data['pending_ocr']      = True
                data['frames_since_ocr'] = 0
                executor.submit(ocr_job, pid, crop)

    # --- Draw bounding boxes ---
    for pid, data in active_plates.items():
        x1, y1, x2, y2 = data['box']
        if data['ocr_done']:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, data['confirmed_label'], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            n = len(data['ocr_buffer'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, f"Reading... {n}/{REQUIRED_FRAMES}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    out.write(np.hstack([frame, sidebar_panel]))
    frame_idx += 1

    if frame_idx % 100 == 0:
        print(f"  Frame {frame_idx}/{total} | tracks: {len(active_plates)} "
              f"| confirmed: {len(confirmed_plates)}")

# -------------------------
# 9. Cleanup
# -------------------------
executor.shutdown(wait=True)
cap.release()
out.release()
print(f"\nDone! {plate_counter} plates saved → {run_dir}")
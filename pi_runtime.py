import cv2
import numpy as np
import onnxruntime as ort
import os
import re
import threading
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# 1. Setup & Config
# -------------------------

DETECT_EVERY_N_FRAMES  = 3      # Run YOLO every N frames
OCR_EVERY_N_FRAMES     = 1      # Submit OCR job every N frames per tracked plate
OCR_WORKERS            = 2      # Parallel OCR threads
REQUIRED_FRAMES        = 3      # Frames needed for consensus
CONSENSUS_THRESHOLD    = 0.66   # 66% of frames must agree on text
MIN_OCR_CONF           = 0.85   # Minimum single-frame OCR confidence to count
YOLO_CONF              = 0.70
YOLO_INPUT_SIZE        = 320
MIN_PLATE_W            = 0
MIN_PLATE_H            = 0
STALE_FRAMES           = 30     # Frames before unmatched track is removed
CONFIRMED_STALE_FRAMES = 5      # Frames before confirmed+saved track is removed
MAX_PENDING_PER_TRACK  = 2      # Max queued OCR futures per track (prevents buildup)

WINDOW_NAME = 'License Plate Detector  —  press Q to quit'

# Filipino plate formats: LLL NNN or LLLL NNNN (no hyphens, no special chars)
# Space is optional since OCR may or may not detect the gap
_PH_PLATE_RE = re.compile(r'^([A-Z]{3,4})\s*(\d{3,4})$')

# Allowlist: letters + digits only — NO hyphen, Filipino plates don't use it
_OCR_ALLOWLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def normalize_plate(raw: str):
    """
    Clean raw OCR text and validate against Filipino plate format.
    Returns canonical 'LLL NNN' / 'LLLL NNNN' string, or None if invalid.
    """
    # Strip everything except alphanumeric (remove accidental hyphens, dots, spaces first)
    clean = ''.join(c for c in raw.upper() if c.isalnum() or c == ' ').strip()
    # Also try without the space in case OCR merged both halves
    compact = ''.join(c for c in clean if c.isalnum())

    for candidate in (clean, compact):
        m = _PH_PLATE_RE.match(candidate.strip())
        if m:
            letters, numbers = m.group(1), m.group(2)
            return f"{letters} {numbers}"   # canonical form with a single space

    return None  # did not match LLL NNN or LLLL NNNN

# -------------------------
# 2. Display environment setup
# -------------------------

def setup_display():
    global HEADLESS
    HEADLESS = False

    if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
        os.environ['DISPLAY'] = ':0'
        print("[DISPLAY] DISPLAY was not set — defaulting to :0")

    try:
        test = np.zeros((2, 2, 3), dtype=np.uint8)
        cv2.imshow('__test__', test)
        cv2.waitKey(1)
        cv2.destroyWindow('__test__')
        print(f"[DISPLAY] OK — using DISPLAY={os.environ['DISPLAY']}")
    except cv2.error:
        print("[DISPLAY] WARNING: cv2.imshow() failed. Running in headless mode.")
        print("          To enable live view, either:")
        print("            • Run locally with a monitor attached")
        print("            • SSH with X forwarding:  ssh -X user@pi")
        HEADLESS = True

setup_display()

# --- OCR backend selection ---
OCR_BACKEND = None

try:
    from rapidocr_onnxruntime import RapidOCR
    # Pass text_score threshold so low-confidence fragments are dropped early
    _rapid_ocr = RapidOCR(text_score=MIN_OCR_CONF)
    OCR_BACKEND = "rapid"
    print("[OCR] Using RapidOCR (fastest)")
except ImportError:
    pass

if OCR_BACKEND is None:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        OCR_BACKEND = "tesseract"
        print("[OCR] Using Tesseract")
    except Exception:
        pass

if OCR_BACKEND is None:
    import easyocr
    _easy_reader = easyocr.Reader(['en'], gpu=False)
    OCR_BACKEND = "easyocr"
    print("[OCR] Using EasyOCR (slowest — consider installing rapidocr-onnxruntime)")

ocr_lock = threading.Lock()

# --- Paths ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs/detect", f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)
output_video_path = os.path.join(run_dir, "output.mp4")
plate_save_dir = os.path.join(run_dir, "plates")
os.makedirs(plate_save_dir, exist_ok=True)

# -------------------------
# 3. YOLO ONNX
# -------------------------

def build_ort_session(model_path):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    print(f"[YOLO] Loaded {model_path} | input: {session.get_inputs()[0].shape}")
    return session

_model_int8 = './runs/detect/license_plate_detector/weights/best_int8.onnx'
_model_fp32 = './runs/detect/license_plate_detector/weights/best.onnx'
ort_session = build_ort_session(_model_int8 if os.path.exists(_model_int8) else _model_fp32)

_input_name  = ort_session.get_inputs()[0].name
_output_name = ort_session.get_outputs()[0].name

def yolo_infer(frame):
    """Run YOLO inference. Returns list of [x1,y1,x2,y2,conf]."""
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = img[np.newaxis]

    raw = ort_session.run([_output_name], {_input_name: img})[0]
    if raw.ndim == 3 and raw.shape[1] == 5:
        raw = raw[0].T
    elif raw.ndim == 3:
        raw = raw[0]

    scale_x = w / YOLO_INPUT_SIZE
    scale_y = h / YOLO_INPUT_SIZE

    boxes = []
    for det in raw:
        if len(det) < 5:
            continue
        cx, cy, bw, bh, conf = det[:5]
        if conf < YOLO_CONF:
            continue
        x1 = int((cx - bw / 2) * scale_x)
        y1 = int((cy - bh / 2) * scale_y)
        x2 = int((cx + bw / 2) * scale_x)
        y2 = int((cy + bh / 2) * scale_y)
        boxes.append([x1, y1, x2, y2, float(conf)])

    if boxes:
        b = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in boxes]
        s = [c for *_, c in boxes]
        idxs = cv2.dnn.NMSBoxes(b, s, YOLO_CONF, 0.45)
        boxes = [boxes[i] for i in (idxs.flatten() if len(idxs) else [])]

    return boxes

# -------------------------
# 4. OCR helpers
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
    """
    Thread-safe OCR.
    Returns (normalized_plate_text, conf) or (None, 0) if text doesn't match
    a valid Filipino plate format (LLL NNN or LLLL NNNN).
    """
    processed = preprocess_for_ocr(plate_img)

    if OCR_BACKEND == "rapid":
        with ocr_lock:
            result, _ = _rapid_ocr(processed)
        if not result:
            return None, 0.0
        # Concatenate all detected text regions in left-to-right order
        # (RapidOCR may split the plate into multiple boxes)
        result_sorted = sorted(result, key=lambda x: x[0][0][0] if x[0] else 0)
        raw_text = ' '.join(r[1] for r in result_sorted if len(r) > 1)
        # Use the minimum confidence across all boxes as the overall conf
        confs = [r[2] for r in result_sorted if len(r) > 2]
        conf = min(confs) if confs else 1.0

    elif OCR_BACKEND == "tesseract":
        import pytesseract
        # psm 7 = single text line (better than psm 8 for plates with a space)
        # Remove hyphen from whitelist — Filipino plates don't use it
        config = (f'--psm 7 --oem 1 '
                  f'-c tessedit_char_whitelist={_OCR_ALLOWLIST}')
        raw_text = pytesseract.image_to_string(processed, config=config).strip()
        conf = 1.0 if raw_text else 0.0

    else:  # easyocr
        with ocr_lock:
            results = _easy_reader.readtext(
                processed,
                allowlist=_OCR_ALLOWLIST,   # no hyphen
                paragraph=False
            )
        if not results:
            return None, 0.0
        # Same as RapidOCR: sort left-to-right, join, take min conf
        results_sorted = sorted(results, key=lambda x: x[0][0][0])
        raw_text = ' '.join(text for _, text, _ in results_sorted)
        conf = min(prob for _, _, prob in results_sorted)

    # --- Validate against Filipino plate format ---
    normalized = normalize_plate(raw_text)
    if normalized is None:
        return None, 0.0
    if conf < MIN_OCR_CONF:
        return None, 0.0

    return normalized, conf

# -------------------------
# 5. Plate tracker / consensus
# -------------------------

class PlateTracker:
    def __init__(self):
        self.tracks      = {}
        self.next_id     = 0
        self.frame_count = 0

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def update(self, detections):
        matched_ids = set()

        for det in detections:
            x1, y1, x2, y2, conf = det
            best_id, best_iou = None, 0.35

            for tid, track in self.tracks.items():
                if track.get('confirmed_text') and track.get('saved'):
                    continue
                iou = self.iou((x1, y1, x2, y2), track['box'])
                if iou > best_iou:
                    best_id, best_iou = tid, iou

            if best_id is not None:
                self.tracks[best_id]['box'] = (x1, y1, x2, y2)
                matched_ids.add(best_id)
            else:
                self.tracks[self.next_id] = {
                    'box': (x1, y1, x2, y2),
                    'ocr_readings': [],
                    'confirmed_text': None,
                    'saved': False,
                    'best_crop': None,
                    'last_seen': self.frame_count,
                    'pending_ocr': 0,
                }
                matched_ids.add(self.next_id)
                self.next_id += 1

        stale = [
            tid for tid, t in self.tracks.items()
            if tid not in matched_ids and (
                (t.get('saved') and self.frame_count - t.get('last_seen', 0) > CONFIRMED_STALE_FRAMES)
                or (self.frame_count - t.get('last_seen', 0) > STALE_FRAMES)
            )
        ]
        for tid in stale:
            del self.tracks[tid]

        for tid in matched_ids:
            if tid in self.tracks:
                self.tracks[tid]['last_seen'] = self.frame_count

        self.frame_count += 1
        return matched_ids

    def add_ocr(self, track_id, text, conf):
        if track_id not in self.tracks:
            return
        t = self.tracks[track_id]
        if t['confirmed_text']:
            return
        t['ocr_readings'].append((text, conf))

        readings = [r[0] for r in t['ocr_readings'] if r[0]]
        if len(readings) >= REQUIRED_FRAMES:
            c = Counter(readings)
            top_text, top_count = c.most_common(1)[0]
            if top_count / len(readings) >= CONSENSUS_THRESHOLD:
                t['confirmed_text'] = top_text
                matching_confs = [r[1] for r in t['ocr_readings'] if r[0] == top_text]
                t['confirmed_conf'] = sum(matching_confs) / len(matching_confs)
                print(f"[TRACK] ID={track_id} confirmed as '{top_text}' "
                      f"with confidence={t['confirmed_conf']:.2f} "
                      f"after {len(readings)} OCR readings")

    def get_track(self, track_id):
        return self.tracks.get(track_id)

# -------------------------
# 6. Drawing helper
# -------------------------

def _draw_and_write(frame, tracker, draw_ids, writer):
    for tid in draw_ids:
        track = tracker.get_track(tid)
        if track is None:
            continue
        x1, y1, x2, y2 = track['box']
        confirmed = track.get('confirmed_text')
        conf = track.get('confirmed_conf', 0.0)
        color = (0, 255, 0) if confirmed else (0, 165, 255)
        label = f"{confirmed} ({conf:.2f})" if confirmed else f"ID:{tid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    writer.write(frame)
    if not HEADLESS:
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt

# -------------------------
# 7. Plate save helper
# -------------------------

def save_plate(plate_text, crop, frame_idx, seen_plates, conf=0.0):
    if plate_text in seen_plates:
        return False
    seen_plates.add(plate_text)
    # Use underscore in filename since the canonical form has a space
    safe_name = plate_text.replace(' ', '_')
    fname = os.path.join(plate_save_dir, f"{safe_name}_{frame_idx}.jpg")
    cv2.imwrite(fname, crop)
    print(f"[PLATE] Detected: {plate_text} | Confidence: {conf:.2f}")
    print(f"[PLATE] Saved image → {fname}")
    return True

# -------------------------
# 8. Main pipeline
# -------------------------

def process_video(input_path):
    if input_path == 0 or (isinstance(input_path, str) and input_path.isdigit()):
        print(f"[INFO] Attempting to open /dev/video0 via V4L2...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        if not cap.isOpened():
            print("[WARN] /dev/video0 failed, trying /dev/video1...")
            cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

        if not cap.isOpened():
            raise RuntimeError("Could not open camera. Try: sudo chmod 666 /dev/video0")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        print("[INFO] Warming up camera sensor...")
        for _ in range(15):
            cap.grab()
    else:
        cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"[INFO] Resolution: {width}x{height} @ {fps:.1f} FPS")
    print(f"[INFO] Recording → {output_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not HEADLESS:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, min(width, 1280), min(height, 720))

    tracker          = PlateTracker()
    ocr_pool         = ThreadPoolExecutor(max_workers=OCR_WORKERS)
    ocr_futures      = {}
    seen_plates      = set()
    frame_idx        = 0
    last_matched_ids = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                detections = yolo_infer(frame)
                detections = [
                    d for d in detections
                    if (d[2] - d[0]) >= MIN_PLATE_W and (d[3] - d[1]) >= MIN_PLATE_H
                ]
                last_matched_ids = tracker.update(detections)
            else:
                last_matched_ids = set()

            # --- Drain completed futures ---
            done = [f for f in list(ocr_futures) if f.done()]
            for f in done:
                tid, saved_crop = ocr_futures.pop(f)
                track = tracker.get_track(tid)
                if track:
                    track['pending_ocr'] = max(0, track.get('pending_ocr', 0) - 1)
                try:
                    text, conf = f.result()
                    if text:
                        tracker.add_ocr(tid, text, conf)
                        track = tracker.get_track(tid)
                        if track:
                            track['best_crop'] = saved_crop
                            if track['confirmed_text'] and not track['saved']:
                                save_plate(track['confirmed_text'], track['best_crop'],
                                           frame_idx, seen_plates, track.get('confirmed_conf', 0.0))
                                track['saved'] = True
                                stale_futures = [pf for pf, (ptid, _) in ocr_futures.items()
                                                 if ptid == tid]
                                for pf in stale_futures:
                                    pf.cancel()
                                    del ocr_futures[pf]
                        else:
                            save_plate(text, saved_crop, frame_idx, seen_plates)
                except Exception as e:
                    print(f"[OCR error] {e}")

            # --- Submit new OCR jobs ---
            if frame_idx % OCR_EVERY_N_FRAMES == 0:
                for tid in last_matched_ids:
                    track = tracker.get_track(tid)
                    if not track or track['confirmed_text']:
                        continue
                    if track.get('pending_ocr', 0) >= MAX_PENDING_PER_TRACK:
                        continue
                    x1, y1, x2, y2 = track['box']
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(width, x2); y2 = min(height, y2)
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size > 0:
                        future = ocr_pool.submit(run_ocr, plate_crop.copy())
                        ocr_futures[future] = (tid, plate_crop.copy())
                        track['pending_ocr'] = track.get('pending_ocr', 0) + 1

            all_active_ids = set(tracker.tracks.keys())
            _draw_and_write(frame, tracker, all_active_ids, writer)
            frame_idx += 1

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        ocr_pool.shutdown(wait=False)
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Video saved → {output_video_path}")
        print(f"[INFO] Plates saved → {plate_save_dir}")

# -------------------------
# 9. INT8 quantization helper (run once)
# -------------------------

def quantize_model(fp32_path='./runs/detect/license_plate_detector/weights/best.onnx'):
    int8_path = fp32_path.replace('.onnx', '_int8.onnx')
    if os.path.exists(int8_path):
        print(f"[QUANTIZE] Already exists: {int8_path}")
        return int8_path
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"[QUANTIZE] Quantizing {fp32_path} → {int8_path} ...")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
    print(f"[QUANTIZE] Done: {int8_path}")
    return int8_path

# -------------------------
# 10. Entry point
# -------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--quantize':
        quantize_model()
        sys.exit(0)

    input_source = int(sys.argv[1]) if (len(sys.argv) > 1 and sys.argv[1].isdigit()) \
                   else (sys.argv[1] if len(sys.argv) > 1 else 0)
    try:
        process_video(input_source)
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
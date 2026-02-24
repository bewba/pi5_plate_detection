import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# 1. Initialize
model = YOLO('./runs/detect/license_plate_detector/weights/best.pt')
reader = easyocr.Reader(['en'], gpu=False)

def deskew_plate(plate_img):
    """Straightens the plate to a front-facing view."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]

            dst = np.array([[0, 0], [300, 0], [300, 100], [0, 100]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            return cv2.warpPerspective(plate_img, M, (300, 100))
    return plate_img

def get_ocr_variants(plate_img):
    """Generates different versions of the image to help OCR."""
    variants = []
    
    # Pre-step: Deskew the plate
    plate_img = deskew_plate(plate_img)
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Variant 1: Adaptive Threshold (Original)
    variants.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    
    # Variant 2: OTSU (Better for plates with background graphics/monuments)
    _, v2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(v2)
    
    # Variant 3: Sharpened + Threshold
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    _, v3 = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    variants.append(v3)

    return variants

def process_single_image(image_path, output_path):
    frame = cv2.imread(image_path)
    if frame is None: return

    results = model(frame, conf=0.6, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            
            # --- RETRY LOGIC START ---
            best_text = ""
            best_prob = 0
            
            # Try each image variant until we hit a high confidence score
            for variant in get_ocr_variants(plate_crop):
                ocr_results = reader.readtext(variant, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
                
                for (_, text, prob) in ocr_results:
                    if prob > best_prob:
                        best_prob = prob
                        best_text = text
                
                if best_prob > 0.90: # Stop early if we have a great result
                    break 
            # --- RETRY LOGIC END ---

            if best_prob >= 0.1: # Display threshold
                display_label = f"{best_text.strip()} ({int(best_prob*100)}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, display_label, (x1, y1 - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(output_path, frame)
    print(f"Processed: {output_path} (Best Confidence: {int(best_prob*100)}%)")

process_single_image('test_image.png', 'output_plate_detected.png')
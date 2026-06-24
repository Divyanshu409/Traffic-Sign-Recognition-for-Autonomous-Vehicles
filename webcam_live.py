import cv2
import json
import numpy as np
from scipy.stats import entropy as scipy_entropy
from tensorflow import keras
import sys
sys.stdout.reconfigure(encoding='utf-8')

CONFIDENCE_THRESHOLD = 70.0
UNCERTAINTY_THRESHOLD = 0.55
ROI_FRACTION = 0.55

model = keras.models.load_model("traffic_sign_recognition_model.keras")

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

MAX_ENTROPY = np.log(len(CLASS_NAMES))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam. Make sure it is connected and not used by another app.")
    exit()

print("Webcam started. Press Q to quit.")
print(f"Settings -> confidence threshold: {CONFIDENCE_THRESHOLD}%, uncertainty threshold: {UNCERTAINTY_THRESHOLD}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    h, w = frame.shape[:2]

    side   = int(min(h, w) * ROI_FRACTION)
    cx, cy = w // 2, h // 2
    x1, y1 = cx - side // 2, cy - side // 2
    x2, y2 = cx + side // 2, cy + side // 2

    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 220, 0), 2)
    cv2.putText(frame, "Hold sign here", (x1 + 4, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)

    img_rgb  = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_r    = cv2.resize(img_rgb, (32, 32)).astype(np.float32) / 255.0
    probs    = model.predict(img_r[np.newaxis], verbose=0)[0]

    top3_indices = np.argsort(probs)[::-1][:3]
    top_idx      = top3_indices[0]
    label        = CLASS_NAMES[top_idx]
    confidence   = probs[top_idx] * 100

    norm_entropy = scipy_entropy(probs) / MAX_ENTROPY

    sign_detected = (confidence >= CONFIDENCE_THRESHOLD and
                     norm_entropy < UNCERTAINTY_THRESHOLD)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if sign_detected:
        cv2.putText(frame, f"Sign : {label}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence : {confidence:.1f}%  |  uncertainty: {norm_entropy:.2f}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        for i, idx in enumerate(top3_indices[1:], start=2):
            cv2.putText(frame,
                        f"  #{i}: {CLASS_NAMES[idx][:30]}  {probs[idx]*100:.1f}%",
                        (10, 65 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "No sign detected", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 60, 255), 2)
        cv2.putText(frame,
                    f"(Best guess: {label}  {confidence:.1f}%  |  uncertainty: {norm_entropy:.2f})",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1)

    cv2.putText(frame, "Press Q to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Traffic Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
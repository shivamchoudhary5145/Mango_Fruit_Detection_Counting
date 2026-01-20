from ultralytics import YOLO
import cv2
import os

# =====================
# CONFIG
# =====================
MODEL_PATH = r"D:\runs\detect\mango_transfer\weights\best.pt"
SOURCE_PATH = r"D:\mango_dataset\valid\images"

CONF = 0.50        # higher confidence to reject leaves
IOU = 0.55
MAX_DET = 30

MIN_AREA_RATIO = 0.01     # reject tiny objects (leaves)
MIN_ASPECT_RATIO = 0.55  # reject thin shapes (leaves)

SAVE_DIR = "runs/detect/mango_clean"

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# LOAD MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# PREDICT
# =====================
results = model.predict(
    source=SOURCE_PATH,
    conf=CONF,
    iou=IOU,
    max_det=MAX_DET,
    stream=True
)

# =====================
# FILTER FUNCTION
# =====================
def is_valid_mango(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    area_ratio = (w * h) / (img_w * img_h)
    aspect_ratio = min(w, h) / max(w, h)

    if area_ratio < MIN_AREA_RATIO:
        return False

    if aspect_ratio < MIN_ASPECT_RATIO:
        return False

    return True

# =====================
# PROCESS RESULTS
# =====================
count = 0

for r in results:
    img = r.orig_img.copy()
    h, w = img.shape[:2]

    if r.boxes is not None:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())

            if not is_valid_mango((x1,y1,x2,y2), w, h):
                continue

            label = f"mango {conf:.2f}"
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(img, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            count += 1

    save_path = os.path.join(SAVE_DIR, os.path.basename(r.path))
    cv2.imwrite(save_path, img)

print("\n✅ Clean prediction completed!")
print(f"📁 Saved to: {SAVE_DIR}")
print(f"🥭 Total mango detections: {count}")

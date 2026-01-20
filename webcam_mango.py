from ultralytics import YOLO
import cv2

# ==============================
# CONFIG
# ==============================
MODEL_PATH = r"D:/runs/detect/mango_transfer/weights/best.pt"
CONF_THRESHOLD = 0.75
IOU_THRESHOLD = 0.45
MAX_DET = 15

# ==============================
# LOAD MODEL
# ==============================
model = YOLO(MODEL_PATH)

# ==============================
# OPEN WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not opened")
    exit()

print("✅ Webcam started. Press Q to quit.")

# ==============================
# LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    mango_count = 0   # ✅ RESET COUNT EVERY FRAME

    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DET,
        device="cpu",
        verbose=False
    )

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            mango_count += 1   # ✅ COUNT MANGO

            label = f"Mango {conf:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ✅ SHOW TOTAL COUNT ON SCREEN
    cv2.putText(frame, f"Mango Count: {mango_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)

    cv2.imshow("Mango Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()

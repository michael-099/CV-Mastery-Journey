from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 prediction without opening a window
    results = model.predict(source=frame, conf=0.3, show=False)

    # Render results onto the frame manually
    annotated_frame = results[0].plot()  # draw bounding boxes and labels
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()
import os
import cv2
import numpy as np

prototxt_path = r"C:\Users\micha\OneDrive\Documents\2017codes\CV\Real_Time_Object_Detection\deploy.prototxt"
model_path = r"C:\Users\micha\OneDrive\Documents\2017codes\CV\Real_Time_Object_Detection\mobilenet_iter_73000.caffemodel"

# Check if model files exist
if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    raise FileNotFoundError("deploy.prototxt or mobilenet_iter_73000.caffemodel not found in the folder!")

# Load pretrained SSD model once
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Open webcam (try 0 first, switch to 1 if 0 doesn't work)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]

    # Prepare input blob for SSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("SSD Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (replace with your path if needed)
model = YOLO("yolov8n.pt")  # or yolov8s.pt, yolov8m.pt, etc.

# Load class names
CLASSES = model.names

# Start video capture
cap = cv2.VideoCapture(1)  # change 0 to your camera index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLOv8 detection
    results = model(frame, imgsz=640, device='cpu')  # use 'cuda' if GPU available

    # results[0].boxes contains bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
    scores = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > 0.25:
            x1, y1, x2, y2 = box.astype(int)
            label = f"{CLASSES[class_id]}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

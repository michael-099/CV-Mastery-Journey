import cv2
import numpy as np

# Load pre-trained model
net = cv2.dnn.readNetFromCaffe(r'C:\Users\micha\OneDrive\Documents\2017codes\CV\Facial_Recognition_and_Landmark_Detection\deploy.prototxt.txt', r'C:\Users\micha\OneDrive\Documents\2017codes\CV\Facial_Recognition_and_Landmark_Detection\res10_300x300_ssd_iter_140000.caffemodel')

# Read image
img = cv2.imread(r'C:\Users\micha\OneDrive\Documents\2017codes\CV\Facial_Recognition_and_Landmark_Detection\face.jpg')
h, w = img.shape[:2]

# Prepare input blob
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)

# Forward pass to get detections
detections = net.forward()

# Draw bounding boxes
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detected Face", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import dlib
from imutils import face_utils
import os
import sys

# Paths to files
image_path = r"C:\Users\micha\OneDrive\Documents\2017codes\CV\Facial_Recognition_and_Landmark_Detection\face.jpg"
predictor_path = r"C:\Users\micha\OneDrive\Documents\2017codes\CV\Facial_Recognition_and_Landmark_Detection\shape_predictor_68_face_landmarks.dat"

# Check if files exist
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    sys.exit()

if not os.path.exists(predictor_path):
    print(f"Error: Predictor file not found at {predictor_path}")
    sys.exit()

# Load image
img = cv2.imread(image_path)
if img is None:
    print("Error: Failed to load image. Check file integrity.")
    sys.exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Initialize dlib detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Detect faces
rects = detector(gray, 1)

# Loop through each detected face
for rect in rects:
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    # Draw each landmark point
    for (x, y) in shape:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

# Show result
cv2.imshow("Facial Landmarks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import face_recognition
import os
import sys

# Full file paths (update the folder path to match your setup)
folder_path = r"C:\Users\micha\OneDrive\Documents\2017codes\CV\Facial_Recognition_and_Landmark_Detection"
known_image_path = os.path.join(folder_path, "person1.jpg")
test_image_path = os.path.join(folder_path, "person2.png")

# Check if files exist
if not os.path.exists(known_image_path):
    print(f"Error: Known image file not found at '{known_image_path}'")
    sys.exit()

if not os.path.exists(test_image_path):
    print(f"Error: Test image file not found at '{test_image_path}'")
    sys.exit()

# Load images
known_image = face_recognition.load_image_file(known_image_path)
test_image = face_recognition.load_image_file(test_image_path)

# Encode faces
try:
    known_encoding = face_recognition.face_encodings(known_image)[0]
except IndexError:
    print(f"Error: No face found in '{known_image_path}'")
    sys.exit()

try:
    test_encoding = face_recognition.face_encodings(test_image)[0]
except IndexError:
    print(f"Error: No face found in '{test_image_path}'")
    sys.exit()

# Compare faces
results = face_recognition.compare_faces([known_encoding], test_encoding)
print("Is the person a match?", results[0])

import os
import sys
import cv2
import face_recognition
import urllib.request

# Folder to store images
folder_path = r"C:\Users\micha\OneDrive\Documents\2017codes\CV\Facial_Recognition_and_Landmark_Detection"
os.makedirs(folder_path, exist_ok=True)

# File paths
known_image_path = os.path.join(folder_path, "person1.jpg")
test_image_path = os.path.join(folder_path, "person2.jpg")  # optional, not used in webcam demo

# Download sample images if missing
if not os.path.exists(known_image_path):
    print("Downloading known face image...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
        known_image_path
    )

if not os.path.exists(test_image_path):
    print("Downloading test face image...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
        test_image_path
    )

# Load known image and encode
known_image = face_recognition.load_image_file(known_image_path)
try:
    known_encoding = face_recognition.face_encodings(known_image)[0]
except IndexError:
    print("No face found in known image.")
    sys.exit()

# Initialize webcam
cap = cv2.VideoCapture(1)  # Use 0 for default webcam, 1 if you have another camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

print("Press 'q' to quit the webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert BGR to RGB for face_recognition
    rgb_frame = frame[:, :, ::-1]

    # Detect faces and encode
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through detected faces
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces([known_encoding], encoding)[0]
        label = "Matched" if match else "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch

use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

cap = cv2.VideoCapture("../Videos/camTraps.mp4")
model = YOLO("../Yolo-Weights/best.pt")

classNames = [
    "Bird",
    "Eastern Gray Squirrel",
    "Eastern Chipmunk",
    "Woodchuck",
    "Wild Turkey",
    "White Tailed Deer",
    "Opossum",
    "Eastern Cottontail",
    "Striped Skunk",
    "Red Fox",
    "Squirrel",
    "Northern Raccoon",
    "Grey Fox",
    "Crow",
    "Chicken",
    "Domestic Cat",
    "Bobcat",
    "Black Bear",
    "Deer"
]

prev_frame_time = 0
new_frame_time = 0

# counter for detected classes
detected_classes = []

# window can be resized
cv2.namedWindow("Wild Cam Detector", cv2.WINDOW_NORMAL)

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    if not success:
        break  # exits when the video ends

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Draw bounding box (always)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), thickness=3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            # Only append to detected_classes if confidence is less than or equal to 0.60
            if confidence <= 0.60:
                detected_classes.append(classNames[cls])

            # Only display text if confidence > 0.55
            if confidence > 0.55:
                cvzone.putTextRect(img, f'{classNames[cls]} {confidence}', (max(0, x1), max(35, y1)), colorT=(0, 0, 0),
                                   colorR=(153, 153, 255), scale=1.5, thickness=2)


    # FPS Calculation
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    cv2.imshow("Wild Cam Detector", img)

    # wait for a key press and check if its esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

# release resources
cap.release()
cv2.destroyAllWindows()

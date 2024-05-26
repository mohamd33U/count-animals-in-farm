from ultralytics import YOLO
import cv2
import math
import cvzone as cz
cap = cv2.VideoCapture('dog.mp4')
#cap.set(3, 640)
#cap.set(4, 480)
model = YOLO('yolov8n.pt')
classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
while True:
    ret, frame = cap.read()  # Correctly assign ret and frame
    # Inside the loop before passing frame to YOLO model


    if not ret:
        break  # Exit the loop if frame retrieval fails

    res = model(frame, stream=True)
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if classes[cls] == 'dog' or classes[cls] == 'cat'  :
                cz.cornerRect(frame, (x1, y1, w, h),colorC=(0, 0, 255), colorR=(0,100, 255))
                cz.putTextRect(frame, f'{classes[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1,
                               colorT=(0, 100, 255),colorR=(10, 10,10))
                print(f"there is a danger i see {classes[cls]}")

    cv2.imshow("frame", frame)

    key = cv2.waitKey(4)
    if key == ord('q'):
        break

cap.release()  # Release the capture device
cv2.destroyAllWindows()
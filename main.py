import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

model = YOLO('yolov8s.pt')

cv2.namedWindow('RGB')

cap = cv2.VideoCapture(0)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

tracker = Tracker()

cy1, cy2 = 322, 368
offset = 6

vh_down = {}
counter = []

vh_up = {}
counter1 = []

L1_start, L1_end = (274, cy1), (500, cy1)
L2_start, L2_end = (177, cy2), (600, cy2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    list = []

    for row in a:
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]
        if 'car' in c or 'person' in c:  # Include 'person' class for human detection
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        if cy1 - offset < cy < cy1 + offset:
            vh_down[id] = time.time()
        if id in vh_down and cy2 - offset < cy < cy2 + offset:
            elapsed_time = time.time() - vh_down[id]
            if id not in counter:
                counter.append(id)
                distance = 10  # meters
                a_speed_kh = (distance / elapsed_time) * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"{int(a_speed_kh)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if cy2 - offset < cy < cy2 + offset:
            vh_up[id] = time.time()
        if id in vh_up and cy1 - offset < cy < cy1 + offset:
            elapsed1_time = time.time() - vh_up[id]
            if id not in counter1:
                counter1.append(id)
                distance1 = 10  # meters
                a_speed_kh1 = (distance1 / elapsed1_time) * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"{int(a_speed_kh1)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1)

    cv2.putText(frame,('L1'),(277,320),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)


    cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
 
    cv2.putText(frame,('L2'),(182,367),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
    d=(len(counter))
    u=(len(counter1))

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
connection.close()
cap.release()
cv2.destroyAllWindows()

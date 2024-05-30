import cv2
from lib.tracking_reid import TrackingReID
from ultralytics import YOLO

cap = cv2.VideoCapture("People HD 1920 1080 25fps.mp4")
model = YOLO("model/yolov8n.pt")
tracking = TrackingReID("bytetrack")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame, verbose = False)
        data = []
        data_dict, track_frame = tracking.track_objs(data, frame, results)
        cv2.imshow('Track-reID', track_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

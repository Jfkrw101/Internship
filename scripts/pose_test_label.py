from ultralytics import YOLO
import cv2
import time

model_path = "model/yolov8n-pose.pt"
model = YOLO(model_path)

prev_time = 0
fps = 0


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

skeleton_map = [
    {'src_kpt_id':15, 'dst_kpt_id':13, 'color':[0, 100, 255], 'thickness':5},       # R.Ankle - R.Knee
    {'src_kpt_id':13, 'dst_kpt_id':11, 'color':[0, 255, 0], 'thickness':5},         # R.Knee - R.Hip
    {'src_kpt_id':16, 'dst_kpt_id':14, 'color':[255, 0, 0], 'thickness':5},         # L.Ankle - L.Knee
    {'src_kpt_id':14, 'dst_kpt_id':12, 'color':[0, 0, 255], 'thickness':5},         # L.Knee - L.Hip
    {'src_kpt_id':11, 'dst_kpt_id':12, 'color':[122, 160, 255], 'thickness':5},     # R.Hip - L.Hip
    {'src_kpt_id':5, 'dst_kpt_id':11, 'color':[139, 0, 139], 'thickness':5},        # R.Shoulder - R.Hip
    {'src_kpt_id':6, 'dst_kpt_id':12, 'color':[237, 149, 100], 'thickness':5},      # L.Shoulder - L.Hip
    {'src_kpt_id':5, 'dst_kpt_id':6, 'color':[152, 251, 152], 'thickness':5},       # R.Shoulder - L.Shoulder
    {'src_kpt_id':5, 'dst_kpt_id':7, 'color':[148, 0, 69], 'thickness':5},          # R.Shoulder - R.Elbow
    {'src_kpt_id':6, 'dst_kpt_id':8, 'color':[0, 75, 255], 'thickness':5},          # L.Shoulder - L.Elbow
    {'src_kpt_id':7, 'dst_kpt_id':9, 'color':[56, 230, 25], 'thickness':5},         # R.Elbow - R.Wrist
    {'src_kpt_id':8, 'dst_kpt_id':10, 'color':[0,240, 240], 'thickness':5},         # L.Elbow - L.Wrist
    {'src_kpt_id':1, 'dst_kpt_id':2, 'color':[224,255, 255], 'thickness':5},        # R.Eyes - L.Eyes
    {'src_kpt_id':0, 'dst_kpt_id':1, 'color':[47,255, 173], 'thickness':5},         # Nose - R.Eyes
    {'src_kpt_id':0, 'dst_kpt_id':2, 'color':[203,192,255], 'thickness':5},         # Nose - L.Ears
    {'src_kpt_id':1, 'dst_kpt_id':3, 'color':[196, 75, 255], 'thickness':5},        # R.Eyes - R.Ears
    {'src_kpt_id':2, 'dst_kpt_id':4, 'color':[86, 0, 25], 'thickness':5},           # L.Eyes - L.Ears
    {'src_kpt_id':3, 'dst_kpt_id':5, 'color':[255,255, 0], 'thickness':5},          # R.Ears - R.Shoulder
    {'src_kpt_id':4, 'dst_kpt_id':6, 'color':[255, 18, 200], 'thickness':5}         # L.Ears - L.Shoulder
]

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    results = model(frame, verbose= False)
    curr_time = time.time()

    
    for r in results:
        keypoints_2d = r.keypoints.xy.numpy()
        keypoints_2d = keypoints_2d[0]

        for idx, keypoint in enumerate(keypoints_2d):
            x, y = int(keypoint[0]), int(keypoint[1])
            if x == 0 and y == 0:
                continue
            print(f"x: {x}, y: {y}")
            frame = cv2.circle(frame, (x, y), 3, (0, 0, 255), 5)
            
            cv2.putText(frame, str(idx), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for skeleton in skeleton_map:
        
            src_kpt_id = skeleton['src_kpt_id']
            src_kpt_x = keypoints_2d[src_kpt_id][0]
            src_kpt_y = keypoints_2d[src_kpt_id][1]
            if src_kpt_x == 0 and src_kpt_y == 0:
                continue
            
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = keypoints_2d[dst_kpt_id][0]
            dst_kpt_y = keypoints_2d[dst_kpt_id][1]
            if dst_kpt_x == 0 and dst_kpt_y == 0:
                continue
            print(f"src_x: {src_kpt_x}, src_y: {src_kpt_y}")
            print(f"dst_kpt_x: {dst_kpt_x}, dst_kpt_y: {dst_kpt_y}")

            skeleton_color = skeleton['color']
        
            skeleton_thickness = skeleton['thickness']
            
            frame = cv2.line(frame, (int(src_kpt_x), int(src_kpt_y)),(int(dst_kpt_x), int(dst_kpt_y)),color=skeleton_color,thickness=skeleton_thickness)

    time_diff = curr_time - prev_time
    fps = 1 / time_diff  

    prev_time = curr_time

    cv2.putText(frame,f"FPS: {fps:.2f}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)
    cv2.imshow('Pose Keypoints', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

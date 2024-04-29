import torch
from ultralytics import YOLO
import time
import cv2


class ObjectDetection:

    def __init__(self, model):
        print("---------- Object Detection Init ----------")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model(model)
        self.frames_count = 0
        self.start_time = time.time()
        
    

    def load_model(self,model):
        model = YOLO(model)  
        model.fuse()
        print(f"Success Load Model\n")
        return model


    def predict(self,frame):
        frame = self.frame
        results = self.model(frame,verbose = False)
        
        return results
    

    def get_data(self, results, frame):
        frame = self.frame
        width, height, _ = frame.shape
        boxes = []
        classes = []
        names = []
        confs = []
        for r in results:
            if self.device == "cuda":
                boxes = r.boxes.xywhn.cuda().tolist()
                classes = r.boxes.cls.cuda().tolist()
                names = r.names
                confs = r.boxes.conf.cuda().tolist()
            else:
                boxes = r.boxes.xywhn.cpu().tolist()
                classes = r.boxes.cls.cpu().tolist()
                names = r.names
                confs = r.boxes.conf.cpu().tolist()


        data_list = []
        for box, classes, conf in zip(boxes, classes, confs):
            x_n, y_n, w_n, h_n = box
            x = int(x_n * width)
            y = int(y_n * height)
            w = int(w_n * width)
            h = int(h_n * height)
            conf = float(conf)
            name = names[int(classes)]

            box_data = {
                "name": name,
                "class": int(classes),
                "confidence": conf,
                "box": [x,y,w,h]
            }
        
            data_list.append(box_data) 


        
        
        return data_list
    
    def detector(self,frame):
        self.frames_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frames_count / elapsed_time 
        
        results = self.predict(frame)
        data_list = self.get_data(results, frame)
        annotated_frame = results[0].plot() 
        annotated_frame = cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        return data_list, annotated_frame, results
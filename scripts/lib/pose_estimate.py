import torch
from ultralytics import YOLO
import time
import cv2


class PoseEstimate:

    def __init__(self, model):
        print("---------- Pose Estimate Init ----------")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model(model)
        self.frames_count = 0
        self.start_time = time.time()
        
    

    def load_model(self, model):
        model = YOLO(model)  
        model.fuse()
        print(f"Success Load Model\n")

    
        return model


    def predict(self,frame):
        frame = self.frame
        results = self.model(frame, verbose = False)
        
        return results
    

    def get_data(self, results, frame):
        frame = self.frame
        width, height, _ = frame.shape
        xyn = []
        data_list = []
        for r in results:
            if self.device == "cuda":
                xyn = r.keypoints.xyn.cuda().tolist()
            else:
                xyn = r.keypoints.xyn.cpu().tolist()


            data_list.append([xyn])
        
        return data_list
    
    def estimator(self,frame):
        self.frames_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frames_count / elapsed_time  
        results = self.predict(frame)
        data_list = self.get_data(results, frame)
        annotated_img = results[0].plot()
        annotated_img = cv2.putText(annotated_img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        return data_list, annotated_img, results 
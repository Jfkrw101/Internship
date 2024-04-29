import cv2
from lib.tracker.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.ops import ltwh2xyxy,xyxy2xywh
import time
class Tracking:

    def __init__(self,cfg_tracker):

        print("---------- Tracking Init ----------")
        cfg_path = "cfg/" + f"{cfg_tracker}.yaml"
        track_cfg = check_yaml(cfg_path)
        cfg = IterableSimpleNamespace(**yaml_load(track_cfg))
        if isinstance(cfg, IterableSimpleNamespace):
            self.tracker = BYTETracker(cfg)  
            print(f"Success Get cfg tracker: {cfg_tracker}\n")
        else:
            raise TypeError("Configuration is not in the expected format. Expected IterableSimpleNamespace.")
        
        self.frames_count = 0
        self.start_time = time.time()


    
    def track_objs(self, data, frame, results):
        annotated_frame = frame.copy()
        self.frames_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frames_count / elapsed_time 

        for result in results:
            outputs = self.tracker.update(result)
            for output in outputs:
                boxes = output[:4]
                track_ids = output[4]
                box = ltwh2xyxy(boxes)
                box = xyxy2xywh(box)
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])


                
                annotated_frame = cv2.rectangle(annotated_frame,(x -( w // 2), y - ( h // 2 )),(w ,h),(0, 0, 255),3)

                annotated_frame = cv2.putText(annotated_frame,f"Track_ID:{track_ids}",(x - w // 2, y - h // 2 - 10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),3)
                annotated_frame = cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        return data, annotated_frame
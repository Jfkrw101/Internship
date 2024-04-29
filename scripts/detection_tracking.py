import cv2
from lib.object_detection import ObjectDetection
from lib.tracking import Tracking
import os
import time
import GPUtil
import argparse
import threading



def detection_tracking(model,tracker,vdo_path, vdo_idx, log_path, output_vdo_path):
    cap = cv2.VideoCapture(vdo_path)
    obj_detection = ObjectDetection(model)
    tracking = Tracking(tracker)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vdo_output_dir = output_vdo_path 
    output_filename = f"detection_tracking_{vdo_idx + 1}.mp4"
    os.makedirs(vdo_output_dir, exist_ok=True)
    output_path = os.path.join(vdo_output_dir, output_filename)
    result_vid = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*"XVID"),fps,size)

    while cap.isOpened:
        success, frame = cap.read()
        if success:
            obj_detection.frame = frame
            data, _, results = obj_detection.detector(frame)
            data_dict, track_frame = tracking.track_objs(data, frame, results)
            result_vid.write(track_frame)
            collect_resource_data(log_path)  
        if not success:
            break

    cap.release()
    result_vid.release()

def multithread_tracking(vdo_dir, model, tracker,log_path, output_vdo_path):
    origin_dir = os.listdir(vdo_dir)
    print(f"Footage Dir contained: {origin_dir}\n")
    threads = []
    for i, vdo_file in enumerate(origin_dir):
        vdo_path = os.path.join(vdo_dir, vdo_file)  
        thread = threading.Thread(target=detection_tracking, args=(model,tracker,vdo_path, i, log_path, output_vdo_path), daemon=True)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()  



def collect_resource_data(log_path):
    os.makedirs(log_path, exist_ok=True)
    cpu_log_path = os.path.join(log_path, "cpu_avg.txt")
    gpu_log_path = os.path.join(log_path, "gpu_mem.txt")
    GPUs = GPUtil.getGPUs()
    gpu_mems = []
    memory_used = 0
    if GPUs:
        gpu = GPUs[0]
        memory_used = gpu.memoryUsed
        memory_total = gpu.memoryTotal
        memory_percent = gpu.memoryUtil * 100
        gpu_mems.append(memory_used)
        gpu_mems.append(memory_total)
        gpu_mems.append(memory_percent)
        print(f"GPU Memory Used: {gpu_mems[0]}/{gpu_mems[1]} MB ({gpu_mems[2]:.2f}%)\n")
    else:
        print("NO GPU DETECTED")

    cpu_avg = os.getloadavg()
    print(f"CPU Average: {cpu_avg}")


    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    with open(cpu_log_path, 'a') as cpu_log:
        cpu_log.write(f"{timestamp}, {cpu_avg[0]:.3f}, {cpu_avg[1]:.3f}, {cpu_avg[2]:.3f}\n")
    with open(gpu_log_path, 'a') as gpu_log:
            gpu_log.write(f"{timestamp}, {gpu_mems}%\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config to Inference VDO")
    parser.add_argument("--model", help = "Path to model", required =True)
    parser.add_argument("--vdo", help = "Path to vdo want to inference", required = True)
    parser.add_argument("--log", help = "Path to log data", required =True)
    parser.add_argument("--saved_dir", help = "Path to save vdo inference", required = True)



    args = parser.parse_args()
    model_path = args.model
    vdo_path = args.vdo
    log_path = args.log
    saved_dir = args.saved_dir

    

    start_time = time.time()
    
    multithread_tracking(vdo_path, model_path, "bytetrack",log_path, saved_dir)

    end_time = time.time()
    time_log_path = os.path.join(log_path, "time_execute.txt")


    execution_time = end_time - start_time  
    print(f"Total execution time: {execution_time:.3f} seconds")
    with open(time_log_path, 'w') as time_log:
        time_log.write(f"{execution_time:.3f}\n")



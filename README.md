# README.md
**For All Work on Internship Period**


## YOLOV8
In This Part Try to Benchmark between Small model and Nano model with Variety Method e.g. Detection, Pose Estimate, Tracking

## Dependency Installation
**main dependency in this project**
```
pip install ultralytics
```

**All Dependency contain in requirement.txt can install follow this**
```
pip3 install -r requirements.txt
```

**Note** All package Install via ultralytics package for more info: [Ultralytics Github](https://github.com/ultralytics/ultralytics)

## Benchmark GPU, CPU, Time Execution

### **Detection**

![](assets/graph/detection.png)

---
### **Detection + Tracking**

![](assets/graph/detection_tracking.png)

---
### **Pose Estimate**

![](assets/graph/pose_estimate.png)

---
### **Pose Estimate + Tracking**

![](assets/graph/pose_tracking.png)

---

## Demo VDO

### **Nano Model**

| **Task**  |**VDO** |
| --------- | --------- |
| **Detection**    |    ![detection gif](assets/demo_vdo/nano/Detection.gif)     |
| **Detection + Tracking**    |    ![detection tracking gif](assets/demo_vdo/nano/Detection_tracking.gif)     |
| **Pose Estimate**    |    ![pose estimate gif](assets/demo_vdo/nano/Pose_Estimate%205.gif)     |
| **Pose Estimate + Tracking**    |    ![pose tracking](assets/demo_vdo/nano/Pose_tracking.gif)     |
---


### **Small Model**

| **Task**  |**VDO**|
| --------- | --------------- |
| **Detection**    |    ![detection gif](assets/demo_vdo/small/detection.gif)     |
| **Detection + Tracking**    |    ![detection tracking gif](assets/demo_vdo/small/detection_tracking.gif)     |
| **Pose Estimate**    |    ![pose estimate gif](assets/demo_vdo/small/pose_estimate.gif)     |
| **Pose Estimate + Tracking**    |    ![pose tracking](assets/demo_vdo/small/pose_tracking.gif)     |



## Multi Class Pose-Detection

### Utility script for label Pose Annotation

**Usage**
```
python3 path/to/script/label_annotated.py --model path/to/model --images_dir path/to/images_dir --label_dir path/to/label_dir --kpt_length number/of/keypoints --class_selected number/of/class/wanted/label --show_img
```
**Parameter**

|**Parameter**|**Usage**|
| --------- | --------------- |
|model|Path to your model (Defualt: yolov8n-pose)|
|images_dir|Path to your image directory|
|label_dir|Path to you label directory|
|kpt_length|Number of keypoints you have (Defualt: 17 keypoints of human)|
|class_selected|Class you want to label it |

---
### Example Image from Utility Scripts


![debug image](assets/annotated_img/anno1.jpg) ![debug image](assets/annotated_img/anno2.jpg) ![debug image](assets/annotated_img/anno3.jpg) ![debug image](assets/annotated_img/anno4.jpg)

--- 
## Example Inference with multi Class Pose-Detection

![multi class pose-detect gif](assets/demo_vdo/multi_pose_detect/safety-pose.gif)


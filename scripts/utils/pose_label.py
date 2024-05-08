from ultralytics import YOLO
import cv2
import os
import random
from skeleton_map import skeleton_map
import torch

os.chdir("obo-mlcv-script/datasets/safety-szm9y_test-h11yv_3")

current_dir = os.getcwd()


test_dir = current_dir + "/test_demo"
train_dir = current_dir + "/train_demo"
valid_dir = current_dir + "/valid_demo"


def get_random_color(class_id):
    random.seed(class_id)  
    return tuple([random.randint(0, 255) for _ in range(3)])


def draw_annotation(images_dir, labels_dir):
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(labels_dir, f'{base_name}.txt')

        if not os.path.exists(label_file):
            print(f"Warning: Label file {label_file} does not exist for image {image_file}. Skipping.")
            continue  

        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        with open(label_file, 'r') as file:
            annotations = file.readlines()  
        for annotation in annotations:
            parts = annotation.strip().split()
            class_id = int(parts[0])
            x_n = float(parts[1])
            y_n = float(parts[2])
            width_n = float(parts[3])
            height_n = float(parts[4])
            

            x_n_px = x_n * width
            y_n_px = y_n * height
            width_px = width_n * width
            height_px = height_n * height

            x = int(x_n_px - (width_px / 2))
            y = int(y_n_px - (height_px / 2))
            w = int(width_px)
            h = int(height_px)
            
            boxes_color = get_random_color(class_id)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), boxes_color, 2)
            text_pos = (x, y - 10)  
            image = cv2.putText(image, f'Class: {class_id}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, boxes_color, 1)
            if len(parts) > 5:
                skeletons = skeleton_map()
                keypoint_data = parts[5:]
                keypoints = []  
                for i in range(0, len(keypoint_data), 2):
                    kpt_x = float(keypoint_data[i]) * width  
                    kpt_y = float(keypoint_data[i + 1]) * height
                    keypoints.append([kpt_x, kpt_y])
                    image = cv2.circle(image, (int(kpt_x), int(kpt_y)), 3, (255, 0, 0), 2)  
                for skeleton in skeletons:
                    start_id = skeleton['src_kpt_id']
                    end_id = skeleton['dst_kpt_id']
                    color = tuple(skeleton['color'])
                    thickness = skeleton['thickness']

                    if start_id < len(keypoints) and end_id < len(keypoints):
                        start_point = keypoints[start_id]
                        
                        end_point = keypoints[end_id]
                        cv2.line(image, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), color, thickness)



        cv2.imshow(f'Image: {image_file}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}")
    model = YOLO("../../../internship/model/yolov8s-pose.pt")
    model.fuse()
    model.to(device)

    return model


def update_annotation(label_comp, class_id=7):
    
    updated_annotations = []

    if not label_comp or not all(len(x) == 2 for x in label_comp):
        print("Error: Invalid label_comp structure")
        return updated_annotations  

    boxes = label_comp[0][0]
    kpt_list = label_comp[0][1]

    new_annotation = f"{class_id} {boxes[0]} {boxes[1]} {boxes[2]} {boxes[3]}"

    for kpt in kpt_list:
        if len(kpt) < 2:
            print("Warning: Keypoint has insufficient data")
            continue
        kpt_x, kpt_y = kpt[0], kpt[1]
        if kpt_x == 0 and kpt_y == 0:
            continue
        new_annotation += f" {kpt_x} {kpt_y}"

    updated_annotations.append(new_annotation)
    return updated_annotations


def label_kpt(images_dir, labels_dir):
    model = get_model()

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        label_file = os.path.join(labels_dir, f'{base_name}.txt')

        if not os.path.exists(label_file):
            print(f"Warning: Label file {label_file} does not exist for image {image_file}. Skipping.")
            continue

        img_path = os.path.join(images_dir, image_file)
        img = cv2.imread(img_path)

        results = model(img, verbose=False)

        label_comp = []
        for r in results:
            boxes_n = r.boxes.xywhn[0].numpy()
            xy_n = r.keypoints.xyn[0].numpy()
            label_comp.append([boxes_n, xy_n])

        with open(label_file, 'r') as file:
            annotations = file.readlines()

        updated_annotations = []
        for annotation in annotations:
            parts = annotation.strip().split()
            class_id = int(parts[0])

            if class_id == 7:  #
                updated_annotations.extend(update_annotation(label_comp, class_id))
            else:
                updated_annotations.append(annotation.strip())

        with open(label_file, 'w') as file:
            for updated_annotation in updated_annotations:
                file.write(updated_annotation + '\n')

        print(f"Updated annotations for {image_file} in {label_file}")
        

# label_kpt(test_dir + "/images", test_dir + "/labels")
# draw_annotation(test_dir + "/images", test_dir + "/labels")
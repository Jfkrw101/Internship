from ultralytics import YOLO
import cv2
import os
import random
from utils.skeleton_map import skeleton_map
import torch

def get_random_color(class_id):
    random.seed(class_id)  
    return tuple([random.randint(0, 255) for _ in range(3)])

def get_bbx(img,boxes, class_id):
    for i,box in enumerate(boxes):
        x, y , w, h = box[0], box[1], box[2], box[3]
        x = int(x - (w / 2))
        y = int(y - (h / 2))
        w = int(w)
        h = int(h)
        color = get_random_color(class_id[i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text_pos = (x, y - 10)  
        img = cv2.putText(img, f'Class: {class_id[i]}', text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img

def get_skeleton(img, kpt_data):
    height, width, _ = img.shape
    keypoints = []
    for i in range(0,len(kpt_data),2):
        kpt_x = float(kpt_data[i]) * width
        kpt_y = float(kpt_data[i + 1]) * height
        print(f"Draw Keypoints: ({kpt_x},{kpt_y})")
        if kpt_x == 0 and kpt_y == 0:
            continue
        keypoints.append([kpt_x, kpt_y])
        img = cv2.circle(img, (int(kpt_x), int(kpt_y)), 3, (255, 0, 0), 2)  
    
    skeletons = skeleton_map()  
    
    for skeleton in skeletons:
        start_id = skeleton['src_kpt_id']
        end_id = skeleton['dst_kpt_id']
        color = tuple(skeleton['color'])
        thickness = skeleton['thickness']

        if start_id < len(keypoints) and end_id < len(keypoints):
            start_point = keypoints[start_id]
            end_point = keypoints[end_id]
            cv2.line(img, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), color, thickness)
    
    return img

def get_model(model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}")
    model = YOLO(model)
    model.fuse()
    model.to(device)
    return model



def draw_annotation(images_dir, labels_dir,show_img = False):
    
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
        boxes = []
        class_ids = []
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

            boxes.append([x_n_px,y_n_px,width_px,height_px])
            class_ids.append(class_id)

            image = get_bbx(image, boxes, class_ids)
            if len(parts) > 5:
                keypoint_data = parts[5:]
                image = get_skeleton(image, keypoint_data)

        if show_img:
            save_dir = os.path.join("internship/debuged_img") 
            os.makedirs(save_dir, exist_ok=True)
            save_dir_path = os.path.join(save_dir, image_file)
            cv2.imwrite(save_dir_path, image)
            cv2.imshow(f'Image: {image_file}', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def label_kpt(images_dir, labels_dir, model, class_selected, kpt_lengths):
    model = get_model(model)

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
            if r.boxes.cls.numpy() != []:          
                xy_n = r.keypoints.xyn[0].numpy()
                label_comp.append(xy_n)
            else:
                continue

        with open(label_file, 'r') as file:
            annotations = file.readlines()

        updated_annotations = []

        kpt_length = kpt_lengths * 2
        for annotation in annotations:
            parts = annotation.strip().split()
            class_id = int(parts[0])
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            if class_id == class_selected:
                if label_comp:
                    default_annotation = f"{class_id} {x} {y} {w} {h}"
                    new_annotation = ""
                    for comp in label_comp:                            
                        for kpt in comp:
                            if len(kpt) < 2:
                                print("Warning: Keypoint has insufficient data")
                                continue
                            kpt_x, kpt_y = kpt[0], kpt[1]
                            new_annotation += f" {kpt_x} {kpt_y}"  
                            print(f"Add kpt: ({kpt_x}, {kpt_y})")

                        updated_annotations.append(default_annotation + new_annotation)
                else:
                    zeros = " ".join(["0"] * kpt_length)
                    default_annotation = f"{class_id} {x} {y} {w} {h} {zeros}"
                    updated_annotations.append(default_annotation)

            else: 
                zeros = " ".join(["0"] * kpt_length)
                default_annotation = f"{class_id} {x} {y} {w} {h} {zeros}"
                updated_annotations.append(default_annotation)

        with open(label_file, 'w') as file:
            for updated_annotation in updated_annotations:
                file.write(updated_annotation + '\n')

        print(f"Updated annotations for {image_file} in {label_file}")

        


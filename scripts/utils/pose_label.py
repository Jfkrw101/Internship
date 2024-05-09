from ultralytics import YOLO
import cv2
import os
import random
from skeleton_map import skeleton_map
import torch

os.chdir("datasets/safety")

current_dir = os.getcwd()


test_dir = current_dir + "/test"
train_dir = current_dir + "/train"
valid_dir = current_dir + "/valid"


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
                    kpt_x = int(kpt_x)
                    kpt_y = int(kpt_y)
                    if kpt_x == 0 and kpt_y == 0:
                        continue
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
    model = YOLO("../../internship/model/yolov8n-pose.pt")
    model.fuse()
    model.to(device)

    return model



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
            if r.boxes.cls.numpy() != []:          
                xy_n = r.keypoints.xyn[0].numpy()
                label_comp.append(xy_n)
            else:
                continue

        with open(label_file, 'r') as file:
            annotations = file.readlines()

        updated_annotations = []

        # Loop through each annotation
        for annotation in annotations:
            parts = annotation.strip().split()
            class_id = int(parts[0])
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Default length for annotations with zeros
            zeros_length = 34  # Adjust as needed

            # If it's class_id 6, ensure label_comp has expected structure
            if class_id == 6:
                if label_comp and len(label_comp[0]) > 0:
                    kpt_list = label_comp[0]  # Extract keypoints

                    # Default annotation with bounding box
                    default_annotation = f"{class_id} {x} {y} {w} {h}"

                    # Create a new annotation with keypoints
                    new_annotation = ""
                    for kpt in kpt_list:
                        if len(kpt) < 2:
                            print("Warning: Keypoint has insufficient data")
                            continue
                        kpt_x, kpt_y = kpt[0], kpt[1]
                        new_annotation += f" {kpt_x} {kpt_y}"  # Append keypoints

                    updated_annotations.append(default_annotation + new_annotation)
                else:
                    # If label_comp is empty or invalid, create a default annotation with zeros
                    zeros = " ".join(["0"] * zeros_length)
                    default_annotation = f"{class_id} {x} {y} {w} {h} {zeros}"
                    updated_annotations.append(default_annotation)

            else:  # If not class_id 6, create a default annotation with zeros
                zeros = " ".join(["0"] * zeros_length)
                default_annotation = f"{class_id} {x} {y} {w} {h} {zeros}"
                updated_annotations.append(default_annotation)



        with open(label_file, 'w') as file:
            for updated_annotation in updated_annotations:
                file.write(updated_annotation + '\n')

        print(f"Updated annotations for {image_file} in {label_file}")
        

# label_kpt(test_dir + "/images", test_dir + "/labels")
# label_kpt(train_dir + "/images", train_dir + "/labels")
# label_kpt(valid_dir + "/images", valid_dir + "/labels")

# draw_annotation(test_dir + "/images", test_dir + "/labels")
# draw_annotation(train_dir + "/images", train_dir + "/labels")
# draw_annotation(valid_dir + "/images", valid_dir + "/labels")
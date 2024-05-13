import argparse
from utils.pose_label import *

def main():
    parser = argparse.ArgumentParser(description="Parameter for Label Annotation")
    parser.add_argument("--model", help="Path to model", required=True)
    parser.add_argument("--images_dir", help="Directory containing images", required=True)
    parser.add_argument("--label_dir", help="Directory containing label .txt", required=True)
    parser.add_argument("--kpt_length", help="Length of keypoint", required=True, default=17, type=int)
    parser.add_argument("--show_img", help="Process to show annotated image for debug", action="store_true")
    parser.add_argument("--class_selected", help="Class ID to label", default=7, type=int)

    args = parser.parse_args()

    model_path = args.model
    img_dir = args.images_dir
    label_dir = args.label_dir
    kpt_len = args.kpt_length
    show_img = args.show_img
    class_id = args.class_selected

    label_kpt(img_dir, label_dir, model_path, class_id, kpt_len)
    if show_img:
        print("Enable Show Draw Image and Save Image")
        draw_annotation(img_dir, label_dir,show_img=show_img)

if __name__ == "__main__":
    main()

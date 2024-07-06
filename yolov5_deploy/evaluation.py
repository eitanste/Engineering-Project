import torch
import cv2
import os
import csv


def get_sub_dirs(root_dir):
    """
    Get a list of subdirectories in the given root directory.

    Args:
    root_dir (str): The path to the root directory.

    Returns:
    list: A list of paths to the subdirectories.
    """
    return [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]


def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s6')
    return model


def analyze_data(model, item_dir_path):
<<<<<<< HEAD
    from yolov5_deploy.deploy import plot_boxes
    from yolov5_deploy.deploy import detectx
=======
    from deploy import plot_boxes
    from deploy import detectx
>>>>>>> e14472f554451fe03d4175ef8bafb379c876da18

    classes = model.names
    item_images_dir_path = os.path.join(item_dir_path, "images")
    item_output_dir_path = os.path.join(item_dir_path, "output")

    # Create output directory if it doesn't exist
    os.makedirs(item_output_dir_path, exist_ok=True)

    images_paths = get_image_paths(item_images_dir_path)
    images_paths.sort()
    output_csv_path = os.path.join(item_dir_path, "detection_results.csv")

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Labels"])

        for img_path in images_paths:
            frame = get_frame(img_path)
            results = detectx(frame, model=model)
            labels = results[0]
            detected_labels = [classes[int(label)] for label in labels]
            writer.writerow([os.path.basename(img_path), ', '.join(detected_labels)])

            # Save the output image with detected labels
            output_image_path = os.path.join(item_output_dir_path, f"output_{os.path.basename(img_path)}")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = plot_boxes(results, frame_bgr, classes=classes)
            cv2.imwrite(output_image_path, frame_bgr)
def get_image_paths(dir_path):
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if is_image_file(f)]


def is_image_file(f):
    return f.lower().endswith(('.png', '.jpg', '.jpeg'))


def get_frame(img_path):
    file_name = os.path.basename(img_path)
    dir_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    print(f"[INFO] Working with  {dir_name}/{file_name}")
    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def main():
<<<<<<< HEAD
    root_dir = r"C:\Users\eitan\OneDrive\Desktop\Master\study\B.A\Year_4\final_project\Engineering-Project\Evaluation"
=======
    root_dir = r"../Evaluation"
>>>>>>> e14472f554451fe03d4175ef8bafb379c876da18
    root_sub_directories = get_sub_dirs(root_dir)
    model = load_model()
    for sub_dir in root_sub_directories:
        analyze_data(model, sub_dir)


if __name__ == '__main__':
    main()

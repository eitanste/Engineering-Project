import time

import cv2
import torch
from ultralytics import YOLO
from deploy import detectx

# Assuming you have functions to load the models and process a frame
def load_yolo(model_size):
    return torch.hub.load('ultralytics/yolov5', model_size)

def load_midas(model_size):
    return torch.hub.load("intel-isl/MiDaS", model_size)

def load_yolo_pose(model_size):
    return YOLO(f'{model_size}.pt')

def process_frame_yolo(model, frame):
    # Placeholder function to process a frame with YOLO model
    results = detectx(frame, model=model)
    return results

def process_frame_midas(model, frame, model_type):
    x = 1
    y = 1
    # Placeholder function to process a frame with Midas model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    # Load the transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # Perform depth estimation
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy array and get the depth value at (x, y)
    depth_map = prediction.cpu().numpy()
    depth_value = depth_map[y, x]

    return depth_value

def process_frame_yolo_pose(model, frame):
    wrist_data = []
    while True:
        results = model(frame)
        for r in results:
            if r.keypoints is not None:
                keypoints = r.keypoints.xy.cpu().numpy()  # Get keypoints' xy coordinates

                for person in keypoints:
                    if person.any():
                        x, y = person[9][0], person[9][1]
                        # print(x, y)
                        wrist_data.append(int(x))
                        wrist_data.append(int(y))
                        cv2.circle(frame, (int(x), int(y)), 7, (0, 255, 0), -1)  # Draw green circles for keypoints
                        wrist_data.append(int(x))
                        wrist_data.append(int(y))
                        cv2.circle(frame, (int(x), int(y)), 7, (0, 255, 0), -1)  # Draw green circles for keypoints

        return wrist_data

# Placeholder frame for processing
frame = cv2.imread(r'C:\Users\eitan\OneDrive\Desktop\Master\study\B.A\Year_4\final_project\Engineering-Project\assets\WhatsApp Image 2024-06-26 at 17.37.19-crop.jpeg')  # Replace with actual image path

model_sizes_yolo = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
model_sizes_midas = ['MiDaS_small', 'DPT_Hybrid', 'DPT_Large']
model_sizes_yolo_pose = ['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose']

results_yolo = []
results_midas = []
results_yolo_pose = []


for model_size in model_sizes_yolo:
    model = load_yolo(model_size)

    start_time = time.time()
    process_frame_yolo(model, frame)
    end_time = time.time()
    processing_time = end_time - start_time
    results_yolo.append((f'YOLOv5-{model_size}', processing_time))
    # results_yolo.append(processing_time)

    print(f'YOLOv5-{model_size}: {processing_time:.4f} seconds')

for model_size in model_sizes_midas:
    model = load_midas(model_size)
    start_time = time.time()
    process_frame_midas(model, frame, model_size)
    end_time = time.time()
    processing_time = end_time - start_time
    results_midas.append((f'Midas-{model_size}', processing_time))
    # results_midas.append(processing_time)
    print(f'Midas-{model_size}: {processing_time:.4f} seconds')

for model_size in model_sizes_yolo_pose:
    model = load_yolo_pose(model_size)
    start_time = time.time()
    process_frame_yolo_pose(model, frame)
    end_time = time.time()
    processing_time = end_time - start_time
    image_path = r'C:\Users\eitan\OneDrive\Desktop\Master\study\B.A\Year_4\final_project\Engineering-Project\assets\WhatsApp Image 2024-06-26 at 17.37.19-crop.jpeg'
    model.predict(image_path, save=True, imgsz=320, conf=0.5)
    results_yolo_pose.append((f'YOLO-Pose-{model_size}', processing_time))
    # results_yolo_pose.append(processing_time)
    print(f'YOLO-Pose-{model_size}: {processing_time:.4f} seconds')

# You can use the results to create a graph using matplotlib
import matplotlib.pyplot as plt

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(model_sizes_yolo, results_yolo, marker='o', label='YOLOv5')
plt.plot(model_sizes_midas, results_midas, marker='s', label='Midas')
plt.plot(model_sizes_yolo_pose, results_yolo_pose, marker='^', label='YOLO-Pose')

plt.xlabel('Model Size')
plt.ylabel('Processing Time (seconds)')
plt.title('Processing Time vs Model Size')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('processing_timessssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss.png')
print(results_yolo)
print(results_midas)
print(results_yolo_pose)





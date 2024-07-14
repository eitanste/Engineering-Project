'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
'''
### importing required libraries
import torch
import cv2
import time
import math
from consts import PERSON, FRAME_SAMPLE_PARAMETER, MIN_DIST_THRESHOLD, dangerous_labels, GREEN_COLOR, RED_COLOR
# from midas_depth_map import get_depth_at_coordinates
# from git telegram_bot import alart_push_notification
from ultralytics import YOLO  # Ensure you have ultralytics package installed
import numpy as np


from deploy_using_v8 import extract_human_keypoints

# Global variables for cooldown
notification_sent = False
cooldown_timer = None
cooldown_duration = 5  # Cooldown duration in seconds


### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx(frame, model):
    frame = [frame]
    results = model(frame)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates


### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, frame_counter, msg, midas, pose_estimation_model, classes):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    global notification_sent, cooldown_timer
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    hazards = dict()
    keypoints = []

    # print(f"[INFO] Total {n} detections. . . ")
    # print(f"[INFO] Looping through all detections. . . ")

    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.33:  ### threshold value for detection. We are discarding everything below this value
            # print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)  ## BBOx coordniates
            text_d = classes[int(labels[i])]

            if text_d in dangerous_labels:
                cv2.rectangle(frame, (x1, y1), (x2, y2), RED_COLOR, 2)  ## BBox
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), RED_COLOR, -1)  ## for text label background
                cv2.putText(frame, text_d + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            elif text_d == PERSON:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN_COLOR, 2)  ## BBox
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), GREEN_COLOR, -1)  ## for text label background

                cv2.putText(frame, text_d + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                keypoints = extract_human_keypoints(pose_estimation_model, frame)

            hazards[text_d] = [x1, y1, x2, y2]
        if frame_counter == FRAME_SAMPLE_PARAMETER:
            is_dangerous_labels = check_dangerous_labels(hazards, keypoints, frame, msg, midas)
            if is_dangerous_labels[0]:
                message = is_dangerous_labels[1]
                print(message)
                if not notification_sent:
                    # alart_push_notification()
                    notification_sent = True
                    cooldown_timer = time.time()
                    break

            else:
                print('NO DANGER DETECTED')

            if notification_sent:
                elapsed_time = time.time() - cooldown_timer
                if elapsed_time >= cooldown_duration:
                    notification_sent = False


    return frame


def check_dangerous_labels(hazards, human_keypoints, frame, msg, midas):
    relevant_keypoints_labels = check_proximity_in_2D(hazards, human_keypoints)
    if relevant_keypoints_labels:
        is_dangerous = measure_distance_in_3D(hazards, human_keypoints,relevant_keypoints_labels,  frame, midas)
        if is_dangerous:
            msg = "WARNING!!!! DANGER DETECTED"
            return True, msg
    return False, msg

def check_proximity_in_2D(hazards, human_keypoints):
    relevant_keypoints_labels = []
    for label in dangerous_labels:
        if is_person_and_hazard_in_one_frame(hazards, label):
            for keypoint in human_keypoints:
                if keypoint[0] != 0 and keypoint[1] != 0 and abs(get_center_of_bbox(hazards[label])[0] - keypoint[0]) < 100 and abs(get_center_of_bbox(hazards[label])[1] - keypoint[1]) < 100:
                    relevant_keypoints_labels.append((keypoint, label))
    return relevant_keypoints_labels





def measure_distance_in_3D_all_keypoints(hazards, relevant_keypoints_labels, frame, midas):
    # Compute the depth map once for the entire frame
    model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid (medium accuracy, medium inference speed)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load the transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Preprocess the image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # Perform depth estimation
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy array
    depth_map = prediction.cpu().numpy()

    for keypoint, label in relevant_keypoints_labels:
        hazard_bbox_center = get_center_of_bbox(hazards[label])
        hazard_depth = depth_map[hazard_bbox_center[1], hazard_bbox_center[0]]
        person_depth = depth_map[int(keypoint[1]), int(keypoint[0])]
        depth_diff = abs(hazard_depth - person_depth)
        if depth_diff < 200:  # Adjust threshold as needed
            return True
    return False

def calculate_weights(hazard_bbox_center, keypoints):
    distances = np.array([np.sqrt((hazard_bbox_center[0] - kp[0])**2 + (hazard_bbox_center[1] - kp[1])**2) for kp in keypoints])
    weights = 1 / (distances + 1e-6)  # Avoid division by zero
    normalized_weights = weights / np.sum(weights)
    return normalized_weights

def measure_distance_in_3D(hazards, human_keypoints, relevant_keypoints_labels, frame, midas):
    # Compute the depth map once for the entire frame
    model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid (medium accuracy, medium inference speed)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load the transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Preprocess the image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # Perform depth estimation
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy array
    depth_map = prediction.cpu().numpy()

    for keypoint, label in relevant_keypoints_labels:
        print("keypoint, label")
        print(keypoint, label)

        hazard_bbox_center = get_center_of_bbox(hazards[label])
        hazard_depth = depth_map[hazard_bbox_center[1], hazard_bbox_center[0]]

        # Calculate weights for each keypoint
        weights = calculate_weights(hazard_bbox_center, human_keypoints)

        # Compute weighted average depth
        weighted_sum = 0
        for kp, weight in zip(human_keypoints, weights):
            weighted_sum += depth_map[int(kp[1]), int(kp[0])] * weight
        weighted_avg_depth = weighted_sum

        depth_diff = abs(hazard_depth - weighted_avg_depth)
        print("hazard_depth")
        print(hazard_depth)

        print("depth_diff")
        print(depth_diff)

        if depth_diff < 50:  # Adjust threshold as needed
            return True
    return False

def is_intersecting(hazards, human_keypoints, label):
    return is_Y_intersect_pe(hazards, human_keypoints,  label) and is_X_intersect_pe(hazards, human_keypoints, label)


def get_center_x_of_bbox(bbox):
    return math.ceil(abs((bbox[0] + bbox[2]) / 2))


def get_center_y_of_bbox(bbox):
    return math.ceil(abs((bbox[1] + bbox[3]) / 2))


def get_center_of_bbox(bbox):
    return [get_center_x_of_bbox(bbox), get_center_y_of_bbox(bbox)]

def is_X_intersect(hazards, label):
    return ((hazards[label][0] >= hazards[PERSON][0] and hazards[label][0] <= hazards[PERSON][2])
            or (hazards[label][2] >= hazards[PERSON][0] and hazards[label][2] <= hazards[PERSON][2]))


def is_X_intersect_pe(hazards, human_keypoints, label):
    for keypoint in human_keypoints:
        if keypoint[0] != 0 and abs(hazards[label][0] - keypoint[0]) < 50:
            print("2d, x axis intersection happened")
            return True
    return False


def is_Y_intersect_pe(hazards, human_keypoints, label):
    for keypoint in human_keypoints:
        if keypoint[1] != 0 and abs(hazards[label][1] - keypoint[1]) < 50:
            print("2d,y axis intersection happened")
            return True
    return False


def measure_distance_in_2D(hazards, human_keypoints, label):
    intersection = is_intersecting(hazards, human_keypoints, label)
    return intersection

def is_person_and_hazard_in_one_frame(hazards, label):
    return (PERSON in hazards.keys() and label in hazards.keys())


### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None, vid_out=None):
    print(f"[INFO] Loading model... ")
    
    depth_model_type = "MiDaS_small"
    object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s6')
    depth_estimation_model = torch.hub.load("intel-isl/MiDaS", depth_model_type)
    pose_estimation_model = YOLO('yolov8x-pose.pt')

    print("loaded the models")

    classes = object_detection_model.names  ### class names in string format

    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detectx(frame, model=object_detection_model)  ### DETECTION HAPPENING HERE
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame, depth_estimation_model, classes=classes)
        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)  ## creating a free windown to show the result
        cv2.imwrite("final_output.jpg", frame)
        while True:
            cv2.imshow("img_only", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                # print(f"[INFO] Exiting. . . ")
                cv2.imwrite("final_output.jpg", frame)  ## if you want to save he output result.
                break
    elif vid_path != None:
        print(f"[INFO] Working with video: {vid_path}")
        ## reading the video
        cap = cv2.VideoCapture(vid_path)
        if vid_out:  ### creating the video writer if video output path is given
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')  ##(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))
        # assert cap.isOpened()
        frame_no = 1
        frame_counter = 0
        msg = ""

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            frame_counter+=1
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=object_detection_model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame, frame_counter, msg, depth_estimation_model, pose_estimation_model, classes=classes)

                cv2.imshow("vid_out", frame)
                if vid_out:
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
                frame_no += 1
                print(frame_counter)

                if frame_counter == 5:
                    frame_counter = 0
        out.release()

        ## closing all windows
        cv2.destroyAllWindows()


EOS_R_CAM = 0
MAIN_WEB_CAM = 2

# DridCam Client
s20FE_WIFI_CONECTION = 3
s20FE_WEB_CONECTION = 'http://192.168.1.129:4747/video'

# main(vid_path="facemask.mp4",vid_out="facemask_result.mp4") ### for custom video
main(vid_path=0,vid_out="knives_tail-out_on_x6.mp4")  # for webcam



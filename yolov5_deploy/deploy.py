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
from midas_depth_map import get_depth_at_coordinates
# from git telegram_bot import alart_push_notification
from ultralytics import YOLO  # Ensure you have ultralytics package installed

from deploy_using_v8 import extract_human_keypoints

# Global variables for cooldown
notification_sent = False
cooldown_timer = None
cooldown_duration = 5  # Cooldown duration in seconds


### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx(frame, model):
    frame = [frame]
    # print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

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
                print(keypoints)

            hazards[text_d] = [x1, y1, x2, y2]
        if frame_counter == FRAME_SAMPLE_PARAMETER:
            is_dangerous_labels = check_dangerous_labels(hazards, keypoints, frame, msg, midas)
            # print(is_dangerous_labels)
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
    for label in dangerous_labels:
        if check_proximity_in_2D(hazards,human_keypoints, label) and check_proximity_in_3D(hazards, human_keypoints, label, frame, midas):
            msg = "WARNING!!!! DANGER DETECTED"
            return True , msg
    return False, msg


def check_proximity_in_2D(hazards, human_keypoints, label):
    return is_person_and_hazard_in_one_frame(hazards, label) and measure_distance_in_2D(hazards, human_keypoints, label)


def check_proximity_in_3D(hazards,  human_keypoints, label, frame, midas):
    # if is_person_and_hazard_in_one_frame(hazards, label):
    #     print("Distance btw obj is " + str((hazards[PERSON][0] / 2) - hazards[label][0] / 2))
    return is_person_and_hazard_in_one_frame(hazards, label) and measure_distance_in_3D(hazards,human_keypoints,  label, frame, midas)


def is_intersecting(hazards, human_keypoints, label):
    return is_Y_intersect_pe(hazards, human_keypoints,  label) and is_X_intersect_pe(hazards, human_keypoints, label)


def get_center_x_of_bbox(bbox):
    return math.ceil(abs((bbox[0] + bbox[2]) / 2))


def get_center_y_of_bbox(bbox):
    return math.ceil(abs((bbox[1] + bbox[3]) / 2))


def get_center_of_bbox(bbox):
    return [get_center_x_of_bbox(bbox), get_center_y_of_bbox(bbox)]


def is_intersecting_3D(hazards, human_keypoints, label, frame, midas):
    hazard_bbox_center = get_center_of_bbox(hazards[label])
    print(human_keypoints)
    for keypoint in human_keypoints:
        print("keypoint")
        print(keypoint)
        hazard_depth = get_depth_at_coordinates(hazard_bbox_center[0], hazard_bbox_center[1], frame, midas)
        person_depth = get_depth_at_coordinates(int(keypoint[0]), int(keypoint[1]), frame, midas)
        depth_diff = abs(hazard_depth - person_depth)
        print(
            f"Depth value hazard ({hazard_bbox_center[0]}, {hazard_bbox_center[1]}): {hazard_depth} \n "
            f"Depth value person ({keypoint[0]}, {keypoint[1]}): {person_depth} \n "
            f"depth_dist: {depth_diff}")
        if depth_diff <100:
            return True
    return False



def is_X_intersect(hazards, label):
    return ((hazards[label][0] >= hazards[PERSON][0] and hazards[label][0] <= hazards[PERSON][2])
            or (hazards[label][2] >= hazards[PERSON][0] and hazards[label][2] <= hazards[PERSON][2]))

#
# def is_Y_intersect(hazards, label):
#     return ((hazards[label][1] >= hazards[PERSON][1] and hazards[label][1] <= hazards[PERSON][3])
#             or (hazards[label][3] >= hazards[PERSON][1] agit nd hazards[label][3] <= hazards[PERSON][3]))kjnk

lkjnkjn
def is_X_intersect_pe(hazards, human_keypoints, label):
    for keypoint in human_keypoints:
        if keypoint[0]!= 0 and abs(hazards[label][0]- keypoint[0])<50:
            print("2d, x axis intersection happened")
            return True
    return False


def is_Y_intersect_pe(hazards, human_keypoints, label):
    for keypoint in human_keypoints:
        if keypoint[1]!= 0 and abs(hazards[label][1]- keypoint[1])<50:
            print("2d,y axis intersection happened")
            return True
    return False


def measure_distance_in_2D(hazards, human_keypoints, label):
    intersection = is_intersecting(hazards, human_keypoints, label)
    return intersection


def measure_distance_in_3D(hazards, human_keypoints, label, frame, midas):
    intersection = is_intersecting_3D(hazards,human_keypoints, label, frame, midas)
    return intersection


def is_person_and_hazard_in_one_frame(hazards, label):
    return (PERSON in hazards.keys() and label in hazards.keys())


### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None, vid_out=None):
    print(f"[INFO] Loading model... ")
    depth_model_type = "DPT_Hybrid"


    object_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')
    depth_estimation_model = torch.hub.load("intel-isl/MiDaS", depth_model_type)
    pose_estimation_model = YOLO('yolov8n-pose.pt')
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
            # by default VideoCapture returns float instead of int
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



# main(vid_path="facemask.mp4",vid_out="facemask_result.mp4") ### for custom video
main(vid_path=0,vid_out="knives_tail-out_on_x6.mp4")  # for webcam



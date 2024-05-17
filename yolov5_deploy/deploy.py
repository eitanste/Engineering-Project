'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
'''
### importing required libraries
import torch
import cv2
import time
import math
import BlynkLib
from yolov5_deploy.consts import PERSON, MIN_DIST_THRESHOLD, dangerous_labels, GREEN_COLOR, RED_COLOR

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
def plot_boxes(results, frame, blynk, classes):
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

    # print(f"[INFO] Total {n} detections. . . ")
    # print(f"[INFO] Looping through all detections. . . ")

    # blynk.run()

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
            ## print(row[4], type(row[4]),int(row[4]), len(text_d))

            hazards[text_d] = [x1, y1, x2, y2]
        # if ('person' in hazards.keys() and lable in hazards.keys()):
        # print('x dist is ' + str(
        #     ('person' in hazards.keys() and lable in hazards.keys()) and (hazards['person'][0] / 2) - hazards[lable][0]/2))
        # print('y dist is ' + str(
        #     (hazards['person'][1] / 2) - hazards[lable][1] / 2))
        #     pass

    if check_dangerous_labels(hazards):
        print('WARNING!!!! DANGER DETECTED')
        if not notification_sent:

            notification_sent = True
            cooldown_timer = time.time()
    else:
        print('NO DANGER DETECTED')

    if notification_sent:
        elapsed_time = time.time() - cooldown_timer
        if elapsed_time >= cooldown_duration:
            notification_sent = False

    return frame


def check_dangerous_labels(hazards):
    for label in dangerous_labels:
        if check_proximity_in_2D(hazards, label):
            return True
    return False


def check_proximity_in_2D(hazards, label):
    # if is_person_and_hazard_in_one_frame(hazards, label):
    #     print("Distance btw obj is " + str((hazards[PERSON][0] / 2) - hazards[label][0] / 2))
    return is_person_and_hazard_in_one_frame(hazards, label) and measure_distance_in_2D(hazards, label)


def is_intersecting(hazards, label):
    return is_Y_intersect(hazards, label) and is_X_intersect(hazards, label)


def is_X_intersect(hazards, label):
    return ((hazards[label][0] >= hazards[PERSON][0] and hazards[label][0] <= hazards[PERSON][2])
            or (hazards[label][2] >= hazards[PERSON][0] and hazards[label][2] <= hazards[PERSON][2]))


def is_Y_intersect(hazards, label):
    return ((hazards[label][1] >= hazards[PERSON][1] and hazards[label][1] <= hazards[PERSON][3])
     or (hazards[label][3] >= hazards[PERSON][1] and hazards[label][3] <= hazards[PERSON][3]))


def measure_distance_in_2D(hazards, label):
    intersection = is_intersecting(hazards, label)
    return intersection


def is_person_and_hazard_in_one_frame(hazards, label):
    return (PERSON in hazards.keys() and label in hazards.keys())


### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None, vid_out=None):
    blynk = init_blynk()

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')
    # model =  torch.hub.load('ultralytics/yolov5', path='last.pt',force_reload=True) ## if you want to download the git repo and then rn #the detection
    # model =  torch.hub.load('/Users/tanyafainstein/Desktop/project/project_yolov5/Engineering-Project', 'custom', source ='local', path='last.pt',force_reload=True) ### The repo is stored locally

    classes = model.names  ### class names in string format

    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detectx(frame, model=model)  ### DETECTION HAPPENING HERE

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame, blynk, classes=classes)

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)  ## creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

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
        notification_sent = False
        cooldown_timer = None
        cooldown_duration = 5

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret:
                # print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame, blynk, classes=classes)

                cv2.imshow("vid_out", frame)
                if vid_out:
                    # print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
                frame_no += 1

        # print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        out.release()

        ## closing all windows
        cv2.destroyAllWindows()


def init_blynk():
    BLYNK_AUTH_TOKEN = "lL47FejJojAm1ZfvU-k6r7WZ64wVebJC"
    blynk = BlynkLib.Blynk(BLYNK_AUTH_TOKEN)
    return blynk


main(vid_path=0, vid_out="default_out.mp4")

    # main(vid_path="facemask.mp4",vid_out="facemask_result.mp4") ### for custom video
         # , vid_out="knives_tail-out_on_x6.mp4")  # for webcam

    # main(img_path="crowd_mask181.jpg") ## for image

'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
'''
from pathlib import Path
import time
import cv2
from enum import Enum
### importing required libraries
import time
import cv2
import torch

from consts import PERSON, dangerous_labels, label_color_mapping
from flask import Flask, Response, jsonify, request
from flask_cors import CORS


#from yolov5_deploy.consts import PERSON, MIN_DIST_THRESHOLD, dangerous_labels, GREEN_COLOR, RED_COLOR
context = (r"C:\Users\eitan\OneDrive\Desktop\fullchain.pem", r"C:\Users\eitan\OneDrive\Desktop\privkey.pem")
app = Flask(__name__)
CORS(app)


ELEMENTS_CONFIG = []
#frame = None
should_play_sound = False

class CameraInput(Enum):
    # cam inputs
    EOS_R_CAM = 0
    MAIN_WEB_CAM = 2
    # DridCam Client
    s20FE_WIFI_CONECTION = 3
    s20FE_WEB_CONECTION = 'http://192.168.1.129:4747/video'


CAMERA_INPUT = CameraInput.EOS_R_CAM.value


### ---CAMERA_INPUT----------------------------------- function to run detection ---------------------------------------------------------
def detectx(frame, model):
    frame = [frame]
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates


### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, classes):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    global should_play_sound
    global ELEMENTS_CONFIG
    import consts
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    hazards = dict()

    dangerous_labels = ELEMENTS_CONFIG

    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.33:  ### threshold value for detection. We are discarding everything below this value
            # print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = get_bbox_corners(row, x_shape, y_shape)  ## BBOx coordniates
            text_d = classes[int(labels[i])]

            color = label_color_mapping.get(text_d)
            if color:
                plot_bbox_by_color(frame, row, text_d, x1, x2, y1, y2, color)

            ## print(row[4], type(row[4]),int(row[4]), len(text_d))

            hazards[text_d] = [x1, y1, x2, y2]


    if check_dangerous_labels(hazards):
        print('WARNING!!!! DANGER DETECTED')
        should_play_sound = True
    else:
        print('NO DANGER DETECTED')
        should_play_sound = False

    return frame


def get_bbox_corners(row, x_shape, y_shape):
    return int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
        row[3] * y_shape)


def plot_bbox_by_color(frame, row, text_d, x1, x2, y1, y2, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  ## BBox
    cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), color, -1)  ## for text label background
    cv2.putText(frame, text_d + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)


def check_dangerous_labels(hazards):
    global ELEMENTS_CONFIG
    for label in ELEMENTS_CONFIG:
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


def main(img_path=None, vid_path=None, vid_out=None, first_run=False, element_config=[]):
    from main_manager import MainManager
    global ELEMENTS_CONFIG
    import consts

    GROUP_CHAT_ID = -4186252810

    manager = MainManager(img_path, vid_path, vid_out, GROUP_CHAT_ID, element_config)
    manager.run(first_run)


def frame_yielder():
    last_frame = None
    while True:
        time.sleep(0.033)  # defines the frame rate
        tmp_frame = Path('tmp.jpg').read_bytes()
        if last_frame != tmp_frame:
            last_frame = tmp_frame
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + tmp_frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    # Return the streaming response
    # global frame
    import consts
    global ELEMENTS_CONFIG

    if consts.frame is not None:
        return Response(frame_yielder(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(main(vid_path=CAMERA_INPUT, vid_out="default_out.mp4", first_run=True, element_config=ELEMENTS_CONFIG),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(main(vid_path=3, vid_out="default_out.mp4"),
    #                 mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/should_play_sound', methods=['GET'])
def play_sound():
    should_play_sound_file = Path('play_sound').read_text().strip().lower() == 'true' if Path('play_sound').exists() else False
    #print(f'Read file {should_play_sound_file}')
    return jsonify({'should_play_sound': should_play_sound_file})

@app.route('/elements_status', methods=['GET'])
def elements_status():
    return jsonify({'already_set': Path('already_set_elements').exists()})

@app.route('/elements', methods=['POST'])
def elements():
    # Return the streaming response
    global ELEMENTS_CONFIG
    ELEMENTS_CONFIG = request.json.get('elements')
    Path('already_set_elements').touch()
    return jsonify({'OK': True})

@app.route('/')
def index():
    # HTML to display the video stream
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Streaming</title>
    </head>
    <body>
        <h1>Video Streaming</h1>
        <img src="/video_feed" width="1080" height="720" />
    </body>
    </html>
    """

if __name__ == "__main__":
    # Path('already_set_elements').unlink() if Path('already_set_elements').exists() else None
    # app.run(host='0.0.0.0', port=5001, ssl_context=context)  # For Production
    # app.run(host='0.0.0.0', port=5001) # For Testing
    main(vid_path=3, vid_out="default_out.mp4")

    # main(vid_path="facemask.mp4",vid_out="facemask_result.mp4") ### for custom video
         # , vid_out="knives_tail-out_on_x6.mp4")  # for webcam

    # main(img_path="crowd_mask181.jpg") ## for image

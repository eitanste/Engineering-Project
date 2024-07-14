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
frame = None
should_play_sound = False

class CameraInput(Enum):
    # cam inputs
    EOS_R_CAM = 0
    MAIN_WEB_CAM = 2
    # DridCam Client
    s20FE_WIFI_CONECTION = 3
    s20FE_WEB_CONECTION = 'http://192.168.1.129:4747/video'

CAMERA_INPUT = CameraInput.MAIN_WEB_CAM.value


### -------------------------------------- function to run detection ---------------------------------------------------------
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
    # global should_play_sound
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
        consts.should_play_sound = True
    else:
        print('NO DANGER DETECTED')
        consts.should_play_sound = False

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


def main(img_path=None, vid_path=None, vid_out=None, first_run=False):
    from main_manager import MainManager
    # global frame
    import consts

    GROUP_CHAT_ID = -4186252810

    manager = MainManager(img_path, vid_path, vid_out, GROUP_CHAT_ID)
    yield from manager.run(first_run)
    # print(f"[INFO] Loading model... ")
    # ## loading the custom trained model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
    # # model =  torch.hub.load('ultralytics/yolov5', path='last.pt',force_reload=True) ## if you want to download the git repo and then rn #the detection
    # # model =  torch.hub.load('/Users/tanyafainstein/Desktop/project/project_yolov5/Engineering-Project', 'custom', source ='local', path='last.pt',force_reload=True) ### The repo is stored locally
    #
    # classes = model.names  ### class names in string format
    #
    # if img_path != None:
    #     print(f"[INFO] Working with image: {img_path}")
    #     consts.frame = cv2.imread(img_path)
    #     consts.frame = cv2.cvtColor(consts.frame, cv2.COLOR_BGR2RGB)
    #
    #     results = detectx(consts.frame, model=model)  ### DETECTION HAPPENING HERE
    #
    #     consts.frame = cv2.cvtColor(consts.frame, cv2.COLOR_RGB2BGR)
    #     consts.frame = plot_boxes(results, consts.frame, classes=classes)
    #
    #     cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)  ## creating a free windown to show the result
    #
    #     while True:
    #         # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    #
    #         cv2.imshow("img_only", consts.frame)
    #
    #         if cv2.waitKey(5) & 0xFF == 27:
    #             # print(f"[INFO] Exiting. . . ")
    #             cv2.imwrite("final_output.jpg", consts.frame)  ## if you want to save he output result.
    #
    #             break
    #
    # elif vid_path != None:
    #     # print(f"[INFO] Working with video: {vid_path}")
    #
    #     ## reading the video
    #     cap = cv2.VideoCapture(vid_path)
    #
    #     if vid_out:  ### creating the video writer if video output path is given
    #
    #         # by default VideoCapture returns float instead of int
    #         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         fps = int(cap.get(cv2.CAP_PROP_FPS))
    #         codec = cv2.VideoWriter_fourcc(*'mp4v')  ##(*'XVID')
    #         out = cv2.VideoWriter(vid_out, codec, fps, (width, height))
    #
    #
    #     # assert cap.isOpened()
    #     frame_no = 1
    #
    #     #cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
    #     while True:
    #         # start_time = time.time()
    #         ret, consts.frame = cap.read()
    #         if ret:
    #             # print(f"[INFO] Working with frame {frame_no} ")
    #
    #             consts.frame = cv2.cvtColor(consts.frame, cv2.COLOR_BGR2RGB)
    #             results = detectx(consts.frame, model=model)
    #             consts.frame = cv2.cvtColor(consts.frame, cv2.COLOR_RGB2BGR)
    #             consts.frame = plot_boxes(results, consts.frame, classes=classes)
    #
    #
    #             ret, buffer = cv2.imencode('.jpg', consts.frame)
    #             consts.frame = buffer.tobytes()
    #
    #             # Use a yield statement to return the frame in a streaming response
    #             if first_run:
    #                 Path('tmp.jpg').write_bytes(consts.frame)
    #                 yield (b'--frame\r\n'
    #                     b'Content-Type: image/jpeg\r\n\r\n' + consts.frame + b'\r\n')
    #             continue
    #             cv2.imshow("vid_out", consts.frame)
    #             if vid_out:
    #                 # print(f"[INFO] Saving output video. . . ")
    #                 out.write(consts.frame)
    #
    #             if cv2.waitKey(5) & 0xFF == 27:
    #                 break
    #             frame_no += 1
    #
    #     # print(f"[INFO] Clening up. . . ")
    #     ### releaseing the writer
    #     out.release()
    #
    #     ## closing all windows
    #     cv2.destroyAllWindows()



def frame_yielder():
    while True:
        time.sleep(0.033)  # defines the frame rate
        tmp_frame = Path('tmp.jpg').read_bytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + tmp_frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    # Return the streaming response
    # global frame
    import consts

    if consts.frame is not None:
        return Response(frame_yielder(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(main(vid_path=CAMERA_INPUT, vid_out="default_out.mp4", first_run=True),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(main(vid_path=3, vid_out="default_out.mp4"),
    #                 mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/should_play_sound', methods=['GET'])
def play_sound():
    # global should_play_sound
    import consts
    return jsonify({'should_play_sound': consts.should_play_sound})

@app.route('/elements_status', methods=['GET'])
def elements_status():
    # global frame
    import consts
    if consts.frame is not None:
        return jsonify({'already_set': True})
    return jsonify({'already_set': False})

@app.route('/elements', methods=['POST'])
def elements():
    # Return the streaming response
    # global ELEMENTS_CONFIG
    import consts
    consts.ELEMENTS_CONFIG = request.json.get('elements')
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
    app.run(host='0.0.0.0', port=5001, ssl_context=context) # For Production
    # app.run(host='0.0.0.0', port=5001) # For Testing
    #main(vid_path=0, vid_out="default_out.mp4")

    # main(vid_path="facemask.mp4",vid_out="facemask_result.mp4") ### for custom video
         # , vid_out="knives_tail-out_on_x6.mp4")  # for webcam

    # main(img_path="crowd_mask181.jpg") ## for image

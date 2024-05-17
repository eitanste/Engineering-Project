'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
'''
import cv2
import torch
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import json

context = (r"C:\Users\eitan\OneDrive\Desktop\fullchain.pem", r"C:\Users\eitan\OneDrive\Desktop\privkey.pem")
app = Flask(__name__)
CORS(app)

ELEMENTS_CONFIG = []

PERSON = 'person'

MIN_DIST_THRESHOLD = 100

# colors for bounding boxes
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)

notification_sent = False
cooldown_timer = None
cooldown_duration = 5  # Cooldown duration in seconds
alert_timer = None  # Timer for keeping the alert on screen

### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx(frame, model):
    frame = [frame]
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

def draw_alert(frame, text="DANGER DETECTED", text_color=(255, 255, 255), thickness=3, font_scale=2, box_color=(0, 0, 255), box_thickness=2, padding=20):
    overlay = frame.copy()
    alpha = 0.6  # Transparency factor

    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    position = (frame.shape[1] // 2 - text_width // 2, frame.shape[0] // 2 + text_height // 2)

    top_left = (position[0] - padding, position[1] - text_height - padding)
    bottom_right = (position[0] + text_width + padding, position[1] + baseline + padding)
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), -1)  # Red background

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, top_left, bottom_right, box_color, box_thickness)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    return frame

def plot_boxes(results, frame, classes):
    global notification_sent, cooldown_timer, alert_timer
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    hazards = dict()

    dangerous_labels = ELEMENTS_CONFIG
    danger_detected = False

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.33:  # Threshold value for detection
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            text_d = classes[int(labels[i])]

            if text_d in dangerous_labels:
                cv2.rectangle(frame, (x1, y1), (x2, y2), RED_COLOR, 2)  # BBox
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), RED_COLOR, -1)  # Text label background
                cv2.putText(frame, text_d + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            elif text_d == PERSON:
                cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN_COLOR, 2)  # BBox
                cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), GREEN_COLOR, -1)  # Text label background
                cv2.putText(frame, text_d + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            hazards[text_d] = [x1, y1, x2, y2]

    if check_dangerous_labels(hazards):
        print('WARNING!!!! DANGER DETECTED')
        danger_detected = True
        if not notification_sent:
            notification_sent = True
            cooldown_timer = time.time()
            alert_timer = time.time()
    else:
        print('NO DANGER DETECTED')

    if notification_sent:
        alert_elapsed_time = time.time() - alert_timer
        if alert_elapsed_time < 3:  # Keep alert on screen for 3 seconds
            frame = draw_alert(frame, text="DANGER DETECTED")
        if alert_elapsed_time >= 3:
            notification_sent = False

    return frame, danger_detected

def check_dangerous_labels(hazards):
    for label in ELEMENTS_CONFIG:
        if check_proximity_in_2D(hazards, label):
            return True
    return False

def check_proximity_in_2D(hazards, label):
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
    global notification_sent, cooldown_timer, alert_timer

    print(f"[INFO] Loading model... ")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')
    classes = model.names  # Class names in string format

    if img_path is not None:
        print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detectx(frame, model=model)  # Detection happening here

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame, danger_detected = plot_boxes(results, frame, classes=classes)

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)  # Creating a free window to show the result

        while True:
            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == 27:
                cv2.imwrite("final_output.jpg", frame)  # Save the output result if needed
                break

    elif vid_path is not None:
        cap = cv2.VideoCapture(vid_path)

        if vid_out:  # Creating the video writer if video output path is given
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')  # (*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        frame_no = 1

        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame, danger_detected = plot_boxes(results, frame, classes=classes)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                if danger_detected:
                    yield (b'--danger\r\n')
                frame_no += 1

        if vid_out:
            out.release()

        cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(main(vid_path=2, vid_out="default_out.mp4"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/elements', methods=['POST'])
def elements():
    global ELEMENTS_CONFIG
    ELEMENTS_CONFIG = request.json.get('elements')
    return jsonify({'OK': True})

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Streaming</title>
    </head>
    <body>
        <h1>Video Streaming</h1>
        <img src="/video_feed" width="1080" height="720" />
        <audio id="alert-sound" src="alert_sound.mp3" preload="auto"></audio>
        <script>
            const source = new EventSource('/video_feed');
            source.addEventListener('danger', function(event) {
                document.getElementById('alert-sound').play();
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, ssl_context=context)

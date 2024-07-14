import cv2
import torch

from notification_manager import NotificationManager
from object_detector import ObjectDetector


class MainManager:
    def __init__(self, img_path=None, vid_path=None, vid_out=None, GROUP_CHAT_ID=None):
        self.img_path = img_path
        self.vid_path = vid_path
        self.vid_out = vid_out
        self.notification_manager = NotificationManager(chat_id=GROUP_CHAT_ID,
                                                        rate_limit_seconds=20,
                                                        consecutive_frames_threshold=6)
        self.detector = ObjectDetector(self.notification_manager)

        ## loading the custom trained model
        print(f"[INFO] Loading model... ")
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
        self.classes = self.model.names  ### class names in string format

    def run(self, first_run):
        if self.img_path != None:
            process_img_input(self.classes, self.img_path, self.model)

        elif self.vid_path != None:
            yield from self.process_vid_input(first_run)

    def capture_frame(self):
        pass

    def process_vid_input(self, first_run=False):
        from pathlib import Path
        import consts

        print(f"[INFO] Working with video: {self.vid_path}")
        ## reading the video
        cap = cv2.VideoCapture(self.vid_path)
        if self.vid_out:  ### creating the video writer if video output path is given
            out = self.init_video_writer(cap)
        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)

        while True:
            # start_time = time.time()
            ret, frame = cap.read()

            if ret:
                consts.frame, hazards = self.process_frame(frame)

                dangerous_interaction = self.detector.check_interactions_with_dangerous_objects(hazards)
                self.notification_manager.check_and_send_notification_if_needed(dangerous_interaction,
                                                                                consts.frame)
                if first_run:
                    # Convert frame to bytes
                    ret, buffer = cv2.imencode('.jpg', consts.frame)
                    if ret:
                        consts.frame = buffer.tobytes()
                        Path('tmp.jpg').write_bytes(consts.frame)
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + consts.frame + b'\r\n')

    def init_video_writer(self, cap):
        # by default VideoCapture returns float instead of int
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v')  ##(*'XVID')
        out = cv2.VideoWriter(self.vid_out, codec, fps, (width, height))
        return out

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.detectx(frame, model=self.model)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame, hazards = self.plot_boxes(results, frame)
        cv2.imshow("vid_out", frame)
        return frame, hazards

    def plot_boxes(self, results, frame):
        """
        --> This function takes results, frame and classes
        --> results: contains labels and coordinates predicted by model on the given frame
        --> classes: contains the strting labels
        """
        from consts import label_color_mapping
        global notification_sent, cooldown_timer

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        hazards = dict()

        ### looping through the detections
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.33:  ### threshold value for detection. We are discarding everything below this value
                # print(f"[INFO] Extracting BBox coordinates. . . ")
                x1, y1, x2, y2 = self.get_bbox_corners(row, x_shape, y_shape)  ## BBOx coordniates
                text_d = self.classes[int(labels[i])]

                color = label_color_mapping.get(text_d)
                if color:
                    self.plot_bbox_by_color(frame, row, text_d, x1, x2, y1, y2, color)

                ## print(row[4], type(row[4]),int(row[4]), len(text_d))

                hazards[text_d] = [x1, y1, x2, y2]

        return frame, hazards

    @classmethod
    def get_bbox_corners(cls, row, x_shape, y_shape):
        return int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
            row[3] * y_shape)

    def plot_bbox_by_color(self, frame, row, text_d, x1, x2, y1, y2, color):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  ## BBox
        cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), color, -1)  ## for text label background
        cv2.putText(frame, text_d + f" {round(float(row[4]), 2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

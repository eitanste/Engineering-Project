import threading

import cv2
import requests
import logging
import time
import numpy
from datetime import datetime, timedelta
from consts import dangerous_labels


class NotificationManager:
    TOKEN = '6837047207:AAEgO2dR4sRu-vyqFhdgcTlOb0blIEdtN5M'
    # CHAT_ID = '@ToddlerGuardbot'
    # GROUP_CHAT_ID = -1002109878950
    FFMPEG_COMMAND = 'ffmpeg -i rtsp://your_camera_stream -f mpegts -c:v mpeg1video -b:v 800k -r 30 -s 640x480 -bf 0 http://your_streaming_server:port'
    BASE_URL = f"https://api.telegram.org/bot{TOKEN}"
    SEND_MESSAGE = "/sendMessage"
    SEND_PHOTO = "/sendPhoto"
    RETRY_LIMIT = 3

    # GROUP_CHAT_ID = -4186252810


    def __init__(self, chat_id, rate_limit_seconds: int = 20, consecutive_frames_threshold: int = 2):
        self.current_frame_number = 0
        self.chat_id = chat_id
        self.notification_sent = False
        self.cooldown_timer = None
        self.consecutive_frames_threshold = consecutive_frames_threshold
        self.rate_limit_seconds = rate_limit_seconds
        self.dangerous_labels = dangerous_labels
        self.last_notification_times = {label: datetime.min for label in dangerous_labels}
        self.consecutive_detections = {label: 0 for label in dangerous_labels}
        self.last_frame_numbers = {label: -1 for label in dangerous_labels}
        self.notification_sent = None
        self.alert_timer = None


    def send_message(self, message: str) -> int:
        payload = {"chat_id": self.chat_id, "text": message}
        for attempt in range(self.RETRY_LIMIT):
            try:
                req = requests.post(self.BASE_URL + self.SEND_MESSAGE, data=payload)
                req.raise_for_status()
                logging.info(f"Message sent successfully: {message}")
                return req.status_code
            except requests.RequestException as e:
                logging.error(f"Failed to send message on attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)
        return req.status_code

    def send_photo(self, frame: numpy.ndarray) -> int:
        file_obj = self.get_image_from_np(frame)
        files = {'photo': file_obj}
        data = {'chat_id': self.chat_id}
        for attempt in range(self.RETRY_LIMIT):
            try:
                res = requests.post(self.BASE_URL + self.SEND_PHOTO, data=data, files=files)
                res.raise_for_status()
                logging.info(f"Photo sent successfully")
                return res.status_code
            except requests.RequestException as e:
                logging.error(f"Failed to send photo on attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)
        return res.status_code

    @staticmethod
    def get_image_from_np(frame):
        from PIL import Image
        from io import BytesIO
        import cv2
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        file_obj = BytesIO()
        image.save(file_obj, format='PNG')
        file_obj.seek(0)
        return file_obj

    def send_notification(self, message: str, frame: numpy.ndarray):
        self.send_message(message)
        if frame is not None:
            self.send_photo(frame)

    def draw_alert(self, frame, text="DANGER DETECTED", text_color=(255, 255, 255), thickness=3, font_scale=2,
                   box_color=(0, 0, 255), box_thickness=2, padding=20):
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
    def check_and_send_notification_if_needed(self, dangerous_interaction, frame):
        self.current_frame_number += 1
        if dangerous_interaction:
            if self.last_frame_numbers[dangerous_interaction] == self.current_frame_number - 1:
                self.consecutive_detections[dangerous_interaction] += 1
            else:
                self.consecutive_detections[dangerous_interaction] = 1

            self.last_frame_numbers[dangerous_interaction] = self.current_frame_number
            current_time = datetime.now()
            last_notification_time = self.last_notification_times[dangerous_interaction]
            if current_time - last_notification_time >= timedelta(seconds=self.rate_limit_seconds):
                if self.consecutive_detections[dangerous_interaction] >= self.consecutive_frames_threshold:
                    print(f"ALERT!!!! Danger was detected: {dangerous_interaction}!")
                    if not self.notification_sent:
                        self.notification_sent = True
                        self.cooldown_timer = time.time()
                        self.alert_timer = time.time()
                    threading.Thread(target=self.send_notification,
                                     args=(f"ALERT!!!! Danger was detected: {dangerous_interaction}!", frame)).start()
                    self.last_notification_times[dangerous_interaction] = current_time
                if self.notification_sent:
                    alert_elapsed_time = time.time() - self.alert_timer
                    if alert_elapsed_time < 3:  # Keep alert on screen for 3 seconds
                        self.draw_alert(frame, text="DANGER DETECTED")
                    if alert_elapsed_time >= 3:
                        self.notification_sent = False
            else:
                print(f"Danger detected too recently for {dangerous_interaction}." +
                      f"Last notification time: {last_notification_time}" +
                      f"Current time: {current_time}" +
                      f"Wait {self.rate_limit_seconds - (current_time - last_notification_time).seconds} seconds.")
        else:
            for label in self.consecutive_detections:
                self.consecutive_detections[label] = 0
                self.last_frame_numbers[label] = -1
            print('NO DANGER DETECTED')

    def alart_push_notification(self, frame):
        res = self.send_message("ALART!!!! Danger was detected!")
        # print(f"The massage got {res}")
        if frame is not None:
            res = self.send_photo(frame)
            # print(f"The massage got {res}")
            

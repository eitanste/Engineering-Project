import requests
import logging
import time
import numpy


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


    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.notification_sent = False
        self.cooldown_timer = None
        self.cooldown_duration = 5

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

    def check_and_send_notification_if_needed(self, dangerous_interaction, frame):
        if dangerous_interaction:
            print('WARNING!!!! DANGER DETECTED')
            if not self.notification_sent:
                self.alart_push_notification(frame)
                self.notification_sent = True
                self.cooldown_timer = time.time()
        else:
            print('NO DANGER DETECTED')

        if self.notification_sent:
            elapsed_time = time.time() - self.cooldown_timer
            if elapsed_time >= self.cooldown_duration:
                self.notification_sent = False

    def alart_push_notification(self, frame):
        res = self.send_message("ALART!!!! Danger was detected!")
        # print(f"The massage got {res}")
        if frame is not None:
            res = self.send_photo(frame)
            # print(f"The massage got {res}")

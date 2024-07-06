import requests
import logging
import time

class NotificationManager:
    TOKEN = '6837047207:AAEgO2dR4sRu-vyqFhdgcTlOb0blIEdtN5M'
    BASE_URL = f"https://api.telegram.org/bot{TOKEN}"
    SEND_MESSAGE = "/sendMessage"
    SEND_PHOTO = "/sendPhoto"
    RETRY_LIMIT = 3

    def __init__(self, chat_id):
        self.chat_id = chat_id

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

    def send_photo(self, frame: 'numpy.ndarray') -> int:
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

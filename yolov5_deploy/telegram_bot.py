# import telegram
# from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
# import requests
# import subprocess
# from flask import Flask, request
import numpy
# # Replace these with your own values
# TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
# CHAT_ID = 'YOUR_CHAT_ID'
# FFMPEG_COMMAND = 'ffmpeg -i rtsp://your_camera_stream -f mpegts -c:v mpeg1video -b:v 800k -r 30 -s 640x480 -bf 0 http://your_streaming_server:port'

# # Set up the Telegram bot
# bot = telegram.Bot(token=TOKEN)

# # Flask app for push notifications
# app = Flask(__name__)

# @app.route('/push', methods=['POST'])
# def push_notification():
#     data = request.json
#     message = data.get('message', 'No message provided')
#     bot.send_message(chat_id=CHAT_ID, text=message)
#     return 'OK', 200

# # Start the FFmpeg stream
# def get_stream(update, context):
#     try:
#         subprocess.Popen(FFMPEG_COMMAND, shell=True)
#         update.message.reply_text("FFmpeg stream started.")
#     except Exception as e:
#         update.message.reply_text(f"Failed to start stream: {e}")

# # Get a snapshot from the camera
# def get_photo(update, context):
#     try:
#         snapshot_url = 'http://your_camera_ip/snapshot.jpg'
#         response = requests.get(snapshot_url)
#         if response.status_code == 200:
#             bot.send_photo(chat_id=update.message.chat_id, photo=response.content)
#         else:
#             update.message.reply_text("Failed to get photo.")
#     except Exception as e:
#         update.message.reply_text(f"Failed to get photo: {e}")

# def main():
#     updater = Updater(token=TOKEN, use_context=True)
#     dispatcher = updater.dispatcher

#     dispatcher.add_handler(CommandHandler('get_stream', get_stream))
#     dispatcher.add_handler(CommandHandler('get_photo', get_photo))

#     updater.start_polling()
#     app.run(host='127.0.0.1', port=5678)

# if __name__ == '__main__':
#     main()



import requests

TOKEN = '6837047207:AAEgO2dR4sRu-vyqFhdgcTlOb0blIEdtN5M'
# CHAT_ID = '@ToddlerGuardbot'
# GROUP_CHAT_ID = -1002109878950
GROUP_CHAT_ID = -4186252810
FFMPEG_COMMAND = 'ffmpeg -i rtsp://your_camera_stream -f mpegts -c:v mpeg1video -b:v 800k -r 30 -s 640x480 -bf 0 http://your_streaming_server:port'
BASE_URL = f"https://api.telegram.org/bot{TOKEN}"
SEND_MESSAGE = "/sendMessage"
SEND_PHOTO = "/sendPhoto"

def send_message(message: str) -> int:
    payload = {"chat_id": GROUP_CHAT_ID, "text": message}
    req = requests.post(BASE_URL + SEND_MESSAGE, data=payload)
    return req.status_code


def send_photo(frame: numpy.ndarray) -> int:

    file_obj = get_image_from_np(frame)

    files = {'photo': file_obj}
    data = {'chat_id': GROUP_CHAT_ID}
    res = requests.post(BASE_URL + SEND_PHOTO, data=data, files=files)

    return res.status_code


def get_image_from_np(frame):
    from PIL import Image
    from io import BytesIO
    import cv2
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    file_obj = BytesIO()
    image.save(file_obj, format='PNG')
    file_obj.seek(0)  # Move to the beginning of the file-like object
    return file_obj


def alart_push_notification(frame):
    res = send_message("ALART!!!! Danger was detected!")
    # print(f"The massage got {res}")
    if frame is not None:
        res = send_photo(frame)
        # print(f"The massage got {res}")

def main():
    alart_push_notification(None)
    return


if __name__ == '__main__':
    main()

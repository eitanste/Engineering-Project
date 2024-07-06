#define BLYNK_TEMPLATE_ID "TMPL6pKoztfwm"
#define BLYNK_TEMPLATE_NAME "Quickstart Template"
#define BLYNK_AUTH_TOKEN "lL47FejJojAm1ZfvU-k6r7WZ64wVebJC"

# define BLYNK_TEMPLATE_ID "TMPL6pKoztfwm"
# define BLYNK_TEMPLATE_NAME "Quickstart Template"
# define BLYNK_AUTH_TOKEN "lL47FejJojAm1ZfvU-k6r7WZ64wVebJC"


"""
Blynk is a platform with iOS and Android apps to control
Arduino, Raspberry Pi and the likes over the Internet.
You can easily build graphic interfaces for all your
projects by simply dragging and dropping widgets.

  Downloads, docs, tutorials: http://www.blynk.cc
  Sketch generator:           http://examples.blynk.cc
  Blynk community:            http://community.blynk.cc
  Social networks:            http://www.fb.com/blynkapp
                              http://twitter.com/blynk_app

This example shows how to perform custom actions
using data from the widget.

In your Blynk App project:
  Add a Slider widget,
  bind it to Virtual Pin V3.
  Run the App (green triangle in the upper right corner)

It will automagically call v3_write_handler.
In the handler, you can use args[0] to get current slider value.
"""

import BlynkLib
import time

# Initialize Blynk
BLYNK_AUTH_TOKEN = "lL47FejJojAm1ZfvU-k6r7WZ64wVebJC"
blynk = BlynkLib.Blynk(BLYNK_AUTH_TOKEN)

# BLYNK_AUTH = 'YourAuthToken'

tmr_start_time = time.time()



while True:
    blynk.run()
    PRE_VAL = 0
    t = time.time()
    if t - tmr_start_time > 6:
        print("sending 'danger notification' to server")
        blynk.virtual_write(0, 1)
        tmr_start_time += 1

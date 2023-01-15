import base64
from io import BytesIO

import cv2
import eventlet
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from tensorflow import keras

sio = socketio.Server()
app = Flask(__name__)

speed_limit = 20


def preprocess_img(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


def send_control(steering_angle, throttle):
    sio.emit('steer', data={'steering_angle':steering_angle.__str__(), 'throttle':throttle.__str__()})


@sio.on('telemetry')
def telemetry(sid, data):
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preprocess_img(image)
    image = np.array([image])
    speed = float(data['speed'])
    throttle = 1.0 - speed/speed_limit
    steering_angle = float(model.predict(image))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("Connected")
    send_control(0, 0)


if __name__ == '__main__':
    model = keras.models.load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

# going around track 1 fully and track 2 stops at second corner



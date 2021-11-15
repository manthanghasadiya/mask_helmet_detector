from django.http.response import HttpResponse
from django.shortcuts import render
# from Train_models.helmet.webcam_helmet_detect import helmet
import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import subprocess

# from detect_mask_video import *


# Create your views here.


def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def mask(request):
    return render(request, 'mask.html')

def helmet(request):
    return render(request, 'helmet.html')


def helmet():
    
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # creating dnn network for face detection
    network = cv2.dnn.readNetFromCaffe(
        'Train_models/helmet/deploy.prototxt.txt', 'Train_models/helmet/res10_300x300_ssd_iter_140000.caffemodel')
    model = keras.models.load_model('Train_models/helmet/helmet.h5')
    start_time = time.time()
    seconds = 10
    while True:
        ret, frame = vid.read()
        if ret:
            # getting height and width od the captured frame
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(
                frame, (300, 300)), 1, (300, 300), (104.0, 177.0, 123.0))
            network.setInput(blob)
            detections = network.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype(int)

                    Y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX-100, startY-100),
                                  (endX+50, endY+50), (0, 0, 255), 2)
                    temp = frame[startY-100:endY+100, startX-100:endX+100]
                    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    temp = cv2.resize(temp, (224, 224))
                    temp = preprocess_input(temp)
                    temp = np.expand_dims(temp, axis=0)
                    pred_val = model.predict(temp)
                    print(pred_val)
                    pred_val = np.ravel(pred_val).item()

                    if pred_val < 0.7:
                        text = 'NO-HELMET' + str(pred_val)
                        cv2.rectangle(frame, (startX-100, startY-100),
                                      (endX+50, endY+50), (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, Y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        text = 'HELMET' + str(pred_val)
                        cv2.rectangle(frame, (startX-100, startY-100),
                                      (endX+50, endY+50), (0, 255, 0), 2)
                        cv2.putText(frame, text, (startX, Y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow('hi', frame)
            cv2.waitKey(10)
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > seconds:
            break

    vid.release()
    cv2.destroyAllWindows()


def helmetDetection(request):
    helmet()
    return render(request, 'home.html')


def mask():
    print("[INFO] loading face detector model...")
    # subprocess.call(['cmd', '/c', 'dir'])
    subprocess.call([r'Y:/Code/mask_helmet_detector/maskDetector/run.bat'])
    return HttpResponse("mask detector")


def maskDetection(request):
    mask()
   	return render(request, 'home.html')

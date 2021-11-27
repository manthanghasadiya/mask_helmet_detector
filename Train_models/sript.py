import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from tensorflow import keras
import numpy as np
import argparse
import imutils
import time
import cv2

#creating dnn network for face detectionY:/Code/mask_helmet_detector/Train_models/helmet/deploy.prototxt.txt
network = cv2.dnn.readNetFromCaffe('Y:/Code/mask_helmet_detector/Train_models/helmet//deploy.prototxt.txt','Y:/Code/mask_helmet_detector/Train_models/helmet/res10_300x300_ssd_iter_140000.caffemodel')
model = keras.models.load_model(
	'Y:/Code/mask_helmet_detector/Train_models/helmet/helmet.h5')
maskNet = load_model(
	"Y:/Code/mask_helmet_detector/Train_models/mask/mask_detector.model")

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)


vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = vid.read()
    if ret:
        # getting height and width od the captured frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        network.setInput(blob)
        detections = network.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype(int)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                temp = frame[startY-100:endY+100, startX-100:endX+100]
                temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                temp = cv2.resize(temp, (224, 224))
                temp = preprocess_input(temp)
                temp = np.expand_dims(temp, axis=0)
                pred_val = model.predict(temp)
                # print(pred_val)
                pred_val = np.ravel(pred_val).item()
                if pred_val < 0.7:
                    text = 'NO-HELMET' + str(pred_val)
                    cv2.rectangle(frame, (startX, startY-10),
                                  (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 3)
                else:
                    text = 'HELMET' + str(pred_val)
                    cv2.rectangle(frame, (startX, startY-10),
                                  (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, Y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 3)

        (locs, preds) = detect_and_predict_mask(frame, network, maskNet)
        for(box, preds) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = preds
            print("mask value ",preds)
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label, (startX, endY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.imshow('hi', frame)
        cv2.waitKey(10)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
vid.release()
cv2.destroyAllWindows()

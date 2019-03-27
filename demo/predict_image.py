from keras.models import model_from_json
import numpy as np
import time
import cv2
import sys
import os

def find_ball(im, model):
   #Load Model
   height,width,_ = im.shape

   #Setup
   x_pred = []
   accuracy = 0.2
   input_size = 128

   #Load Image

   im = cv2.resize(im, (input_size, input_size))
   x_pred.append(im)
   x_pred = np.array(x_pred, np.float)/255

   #Create Prediction
   prediction = model.predict(x_pred)
   prediction2 = cv2.resize(prediction[0], (width,height))

   prediction2 = np.array(prediction2)
   prediction3 = np.power(prediction2,2) * 255
   finalIm = cv2.cvtColor(prediction3, cv2.COLOR_GRAY2BGR)

   return finalIm

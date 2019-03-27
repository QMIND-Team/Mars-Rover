from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import sys
import os

def main(filename):
    #Load Model
    json_file = open('nn_struct.json', 'r')
    model = json_file.read()
    json_file.close()
    model = model_from_json(model)
    model.load_weights('best_weights_256.hdf5')


    #Setup
    x_pred = []
    accuracy = 0.9
    input_size = 128

    #Load Image
    im = cv2.imread(filename)
    im = cv2.resize(im, (input_size, input_size))
    x_pred.append(im)
    x_pred = np.array(x_pred, np.float)/255

    #Create Prediction
    prediction = model.predict(x_pred)
    prediction2 = cv2.resize(prediction[0], (350,350))

    prediction2 = np.array(prediction2)
    prediction3 = np.power(prediction2,20) * 255
    finalIm = cv2.cvtColor(prediction3, cv2.COLOR_GRAY2BGR)

    #Save Image
    cv2.imwrite('mask.jpg',finalIm)
    plt.imshow(prediction3, cmap = 'gray', interpolation = 'bicubic')
    plt.show()

main('test2.jpg')

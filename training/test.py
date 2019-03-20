from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import sys
import os

MAIN_DIR = os.getcwd()
INPUT_DIR = os.path.join(MAIN_DIR,'input')
TRAIN_DIR = os.path.join(INPUT_DIR,'train')
MASK_DIR = os.path.join(INPUT_DIR,'mask')

#Load Model
json_file = open('nn_struct.json', 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights('best_weights_256.hdf5')

counter = 1
correctPix = 0
truePix = 0
for filename in os.listdir(TRAIN_DIR):
    #Setup
    x_pred = []
    accuracy = 0.9
    input_size = 128

    #Load Image
    path = os.path.join(TRAIN_DIR,filename)
    im = cv2.imread(path)
    im = cv2.resize(im, (input_size, input_size))
    x_pred.append(im)
    x_pred = np.array(x_pred, np.float)/255

    #Create Prediction
    prediction = model.predict(x_pred)
    prediction2 = cv2.resize(prediction[0], (128,128))
    prediction2 = np.array(prediction2)
    prediction3 = np.power(prediction2,20) * 255
    
    path = os.path.join(MASK_DIR,filename)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    for i in range(len(prediction3)):
        for j in range(len(prediction3[i])):
            if mask[i][j] == 0 :
                truePix += 1
                if prediction3[i][j] < 1:
                    correctPix += 1
    
    print(counter)
    counter +=1

print("The accuracy is: ", correctPix/truePix)



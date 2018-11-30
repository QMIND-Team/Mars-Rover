# -*- coding: utf-8 -*-
"""
@author: James Xie
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('testimage.jpg')
plt.imshow(img)

def Preprocess(img, ball):
    #Change colour space to HSV
    #Note the image must originally be in BGR format. Depending on the image source, they may originally be in RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ball = cv2.cvtColor(ball, cv2.COLOR_BGR2HSV)
    
    #perform colour white re-balance


    #sharpness equalization
    ImgSharp = cv2.Laplacian(img[:,:,2], cv2.CV_64F).var()
    BallSharp = cv2.Laplacian(ball[:,:,2], cv2.CV_64F).var()
    
    while abs(1 - BallSharp/ImageSharp) > 0.05:
        if ImgSharp > BallSharp:
            
        else:
        
    
    #Intensity histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img[:,:,2] = clahe.apply(img[:,:,2])

    #img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    #crop to square and resize to 256 x 256
    h = min(img.shape[:2])
    img = cv2.resize(img[0:h,0:h], (256,256))
    
    return img

Preprocess(img, img)
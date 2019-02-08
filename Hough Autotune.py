# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:10:40 2019

@author: James Xie
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def circleFind():
    img = cv2.imread('Test.jpg',0)
    imgcopy = img
    
    if len(img.shape) == 3:
        #grayscale the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.medianBlur(img,5)
    _, img = cv2.threshold(img,250,255,cv2.THRESH_BINARY)
    
    #guess parameters
    p2 = 150 #2nd parameter for the Hough Circle algorithm
    guessR = max(img.shape)/2 #maximum possible radius of the ball
    n = 1 #number of expected balls

    circles = None
    breakout = False #variable used for debugging to get out of the while loops
    
    #############################################################
    #       MAIN CODE
    #############################################################
    
    #The autotune begins at a high roundness and largest possible radius and attempts to find the circle
    #If it cannot find one, it will decrease the radius
    #If it cannot find any, it will relax the roundness slightly and then begin again
    
    while p2 > 1 and breakout == False:
        #start with the most precise circle
        guess_dp = 1
        while guess_dp < 9 and breakout == False:
            guessR = max(img.shape)/2
            while guessR > 20:
                circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp=guess_dp, minDist=max(img.shape[1], img.shape[0])/10,
                                param1=50,param2=p2,minRadius=(guessR - 3),maxRadius=(guessR + 3))
                
                if circles is not None and len(circles[0]) == n:
                    breakout = True
                    break
                
                guessR = guessR - 5
            guess_dp = guess_dp + 1
        p2 = p2 - 2
        
    return circles[0], imgcopy

####################################Code to visualize what it found

circles, imgcopy = circleFind()

output = np.copy(imgcopy)
x, y, r = np.round(circles[0]).astype("int")
cv2.circle(output, (x,y), r, (0,0,255), 10)
cv2.imshow('meep',np.hstack([imgcopy, output]))
cv2.waitKey(0)

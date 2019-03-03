from Preprocessing import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import os

MAIN_DIR = os.getcwd()
BALLFOLDER = os.path.join(MAIN_DIR,'balls')
BACKGROUNDFOLDER = os.path.join(MAIN_DIR,'backgrounds')
NEWFOLDER = os.path.join(MAIN_DIR,'train')
MASKFOLDER = os.path.join(MAIN_DIR,'masks')

counter = 1
for ball in os.listdir(BALLFOLDER):
    for bg in os.listdir(BACKGROUNDFOLDER):
        path = os.path.join(BALLFOLDER,ball)
        ball_img = cv2.imread(path,1)
        path = os.path.join(BACKGROUNDFOLDER,bg)
        bg_img = cv2.imread(path,1)

        cir = circleFind(ball_img)
        ball_img = ballCrop(ball_img,cir)
        bg_img = bgCrop(bg_img)
        ball = adjust_brightness(ball_img, bg_img)
        finalIm, mask = ball_embed(ball_img, bg_img)
        
        path = os.path.join(NEWFOLDER,'{}.png'.format(counter))
        cv2.imwrite(path,finalIm)

        path = os.path.join(MASKFOLDER,'mask{}.png'.format(counter))
        cv2.imwrite(path,mask)

        counter += 1

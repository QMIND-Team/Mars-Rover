from Preprocessing import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import os

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(MAIN_DIR, '..'))
TRAINING_DIR = os.path.abspath(os.path.join(PARENT_DIR,"training"))
BALLFOLDER = os.path.join(MAIN_DIR,'balls')
BACKGROUNDFOLDER = os.path.join(MAIN_DIR,'backgrounds')
NEWFOLDER = os.path.join(TRAINING_DIR,'train')
MASKFOLDER = os.path.join(TRAINING_DIR,'masks')

counter = 1
otherCount = 1
for ball in os.listdir(BALLFOLDER):
   print("Working on Ball {} ...".format(otherCount))
   path = os.path.join(BALLFOLDER,ball)
   ball_img = cv2.imread(path,1)
   cir = circleFind(ball_img)
   usableBall_img = ballCrop(ball_img,cir)

   for bg in os.listdir(BACKGROUNDFOLDER):
       path2 = os.path.join(BACKGROUNDFOLDER,bg)
       bg_img = cv2.imread(path2,1)

       bg_img = bgCrop(bg_img)

       ball = adjust_brightness(usableBall_img, bg_img)
       finalIm, mask = ball_embed(ball, bg_img)

       path = os.path.join(NEWFOLDER,'{}.png'.format(counter))
       cv2.imwrite(path,finalIm)

       path = os.path.join(MASKFOLDER,'mask{}.png'.format(counter))
       cv2.imwrite(path,mask)

       counter += 1
   otherCount += 1
   

"""
1. Load images
  Return ball and background (bg) image
2. Locate ball
  Return ball location
3. Crop to ball
  Return cropped ball image
4. Crop bg to square
  Return cropped bg image
5. Adjust background (bg) and ball image properties
  Input ball, bg
  Return ball
6. Embed ball into bg & scale to 128x128
  Input ball, bg
  Return combined, mask
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import math

def circleFind(img):
    #if colour, then grayscale the image   
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.medianBlur(img,5)
    _, img = cv2.threshold(img,250,255,cv2.THRESH_BINARY)
    
    n = 1 #number of expected balls
    circles = None
    breakout = False
    
    #The autotune begins at a high roundness and largest possible radius and attempts to find the circle
    #If it cannot find one, it will decrease the radius
    #If it cannot find any, it will relax the roundness slightly and then begin again
    
    p2 = 150 #2nd parameter for the Hough Circle algorithm
    while p2 > 1 and breakout == False:
        guess_dp = 1 #start with the most precise circle
        while guess_dp < 9 and breakout == False:
            guessR = np.round(max(img.shape)/2).astype("int")
            while guessR > 20 and breakout == False:
                circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp=guess_dp, minDist=max(img.shape[1], img.shape[0])/10,
                                param1=50,param2=p2,minRadius=(guessR - 3),maxRadius=(guessR + 3))
                
                if circles is not None and len(circles[0]) == n:
                    breakout = True
                
                guessR -= 5
            guess_dp += 1
        p2 -= 2
        
    return circles[0][0]
    
def ballCrop(img, cir):
    x,y,r = np.round(cir).astype("int")
    img = img[y-r:y+r,x-r:x+r]
    return img
    
def bgCrop(img):
    s = min(img.shape[0:1])
    img = img[0:s,0:s]
    return img
    
def img_bright(img):
    #Luma to calculate average brightness
    b, g, r = cv2.split(img)
    img_w, img_h = b.shape
    img_avg_bright = sum(map(sum, 0.299*r + 0.587*g + 0.114*b))/(img_w*img_h)
    
    return img_avg_bright
    
def adjust_brightness(ball_img, bg_img):
    ball_avg_brightness = img_bright(ball_img)
    bg_avg_brightness = img_bright(bg_img)
    
    #Determine which img is brightest, and adjust brightness based on that
    #(Not sure conversion between picture brightness vs. beta of transformation)
    #(assuming adding a beta based on difference in avg. Luma will suffice)
    
    B = ball_avg_brightness - bg_avg_brightness
    ball_img = cv2.convertScaleAbs(ball_img, alpha=1, beta=-B)
    
    return ball_img

def ball_embed(ball_img, bg_img):
    #Accepts a colour image containing a tennis ball with a white background and embeds the tennis ball into a different background
    #Outputs both the new background image with the tennis ball and a mask

    #Convert background white pixels to black, then erode to remove white edge
    kernel = np.ones((5,5),np.uint8)
    gray = cv2.cvtColor(ball_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
    ball_img[thresh == 255] = 0
    ball_img = cv2.erode(ball_img,kernel,iterations = 1)
    
    #Scale and rotate the ball by random amounts
    rescale = np.random.random()
    rotate = 360.0*np.random.random()
    ball_img = cv2.resize(ball_img, (0,0), fx=rescale, fy=rescale, interpolation = cv2.INTER_CUBIC)
    rows,cols,_ = ball_img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate,1)
    ball_img = cv2.warpAffine(ball_img,M,(cols,rows))
    
    #Embed ball randomly into background
    #Note: this doesn't allow the ball to ever be partially shown on left or top edges of bg_img
    h,w,_ = bg_img.shape
    orig_rows, orig_cols,_ = ball_img.shape
    offsetY = np.random.randint(0, h)
    offsetX = np.random.randint(0, w)

    end_rows = rows
    end_cols = cols

    if (offsetY + orig_rows > h):
      end_rows = orig_rows - (orig_rows + offsetY - h)
    if (offsetX + orig_cols > w):
      end_cols = orig_cols - (orig_cols + offsetX - w)

    mask = np.zeros(h*w*3)
    mask = np.reshape(mask, (h,w,3))
    for i in range(rows):
        for j in range(cols):
            if offsetY+i<h and offsetX+j<w:
                if math.sqrt((i-rows/2)**2 + (j-cols/2)**2)<rows/2-rows/10:
                    bg_img[offsetY+i][offsetX+j] = ball_img[i][j]
                    mask[offsetY+i][offsetX+j][0] = 255
                    mask[offsetY+i][offsetX+j][1] = 255
                    mask[offsetY+i][offsetX+j][2] = 255
                
    #bg_img[offsetY:offsetY+end_rows,offsetX:offsetX+end_cols,:] = ball_img[0:end_rows, 0:end_cols,:]

    #If the offset + location of the ball > height, do some sort of conditional setting of pixels
    #mask = mask_Maker(bg_img, ball_img, offsetY, offsetX, end_rows, end_cols)
    bg_img = cv2.resize(bg_img, (128, 128), interpolation = cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (128, 128), interpolation = cv2.INTER_CUBIC)
    return bg_img, mask

def mask_Maker(bg_img, ball_img, offsetY, offsetX, end_rows, end_cols):
    blank = np.zeros(bg_img.shape, dtype=np.uint8)
  
    blank[offsetY:offsetY+end_rows,offsetX:offsetX+end_cols,:] = ball_img[0:end_rows, 0:end_cols, :]
    blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    _, ball_mask = cv2.threshold(blank,1,255,cv2.THRESH_BINARY)
    ball_mask = cv2.resize(ball_mask, (128, 128), interpolation = cv2.INTER_CUBIC)
    
    return ball_mask



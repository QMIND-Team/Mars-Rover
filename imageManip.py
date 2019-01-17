import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

#New code proposed to replace existing (Existing still shown below)
#These functions accept an n*m*3 BGR-colour image containing a white background with a tennis ball located inside
#The functions will extract the ball from the image and embed it into a different 128*128 background at some randomized location and orientation

def ball_detector(ball_img):
    #Take a ball image
    #Detect ball using hough circle
    #use centerpoint and radius to crop a square around ball
    #scale square to 128*128
    
    
    return crop_ball_img

def mask_generator(crop_ball_img, background_img):
    #Accepts a 128*128*3 image containing a tennis ball with a white background and embeds the tennis ball into a different background
    #Outputs both the new background image with the tennis ball and a mask
    
    #Convert white pixels to black, then erode to remove white edge
    kernel = np.ones((5,5),np.uint8)
    gray = cv2.cvtColor(crop_ball_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    crop_ball_img = crop_ball_img - mask
    
    #Scale and rotate the ball by random amounts
    rescale = np.random.random
    rotate = 360*np.random.random
    crop_ball_img = cv2.resize(crop_ball_img, fx=rescale, fy=rescale, interpolation = cv2.INTER_CUBIC)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate,1)
    crop_ball_img = cv2.warpAffine(crop_ball_img,M,(128,128))
    
    #Embed ball randomly into an empty 128*128 matrix
    blank = np.array([128,128,3], dtype = float)
    h,w,_ = crop_ball_img.shape
    offsetY = np.random.random_integers(0, 128)
    offsetX = np.random.random_integers(0, 128)
    blank[offsetY:offsetY+h,offsetX:offsetX+w] = crop_ball_img
    
    #Generate mask
    gray_blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    ball_mask = cv2.threshold(gray_blank,1,255,cv2.THRESH_BINARY)
    
    #With background, subtract mask, then add ball
    embedded_img = (background_img - cv2.cvtColor(ball_mask, cv2.COLOR_GRAY2BGR)) + blank
    
    return embedded_img, ball_mask


################################################################
def masker(img,accuracy):
    mask = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j][0] > accuracy and img[i][j][1] > accuracy and img[i][j][2] > accuracy:
                mask[i][j][0] = 0
                mask[i][j][1] = 0
                mask[i][j][2] = 0
            else:
                mask[i][j][0] = 255
                mask[i][j][1] = 255
                mask[i][j][2] = 255
    return mask

def cleaner(img,accuracy):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j][0] > accuracy  and img[i][j][1] > accuracy and img[i][j][2] > accuracy:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
    return img

def scaler(img, sF, **kwargs):
    h, w = img.shape[:2]

    zoom_tuple = (sF,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if sF < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * sF))
        zw = int(np.round(w * sF))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif sF > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / sF))
        zw = int(np.round(w / sF))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    else:
        out = img
    return out

def rotater(img, rF):
    out = ndimage.rotate(img,rF)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if out[i][j][0] == 0  and out[i][j][1] == 0 and out[i][j][2] == 0:
                out[i][j][0] = 255
                out[i][j][1] = 255
                out[i][j][2] = 255
    return out

def translator(img, xcord, ycord):
    pass


def slicer(img, mask):
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j][0] == 255  and mask[i][j][1] == 255 and mask[i][j][2] == 255:
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
    return img

def adder(img1,img2):
    for i in range(len(img1)):
        for j in range(len(img1[i])):
            if img1[i][j][0] == 255  and img1[i][j][1] == 255 and img1[i][j][2] == 255:
                img1[i][j][0] = img2[i][j][0]
                img1[i][j][1] = img2[i][j][1]
                img1[i][j][2] = img2[i][j][2]
    return img1
    


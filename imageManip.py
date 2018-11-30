import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

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
    


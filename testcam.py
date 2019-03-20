import cv2
import time
from predict_image import *

def show_webcam(mirror=False):
    json_file = open('nn_struct.json', 'r')
    model = json_file.read()
    json_file.close()
    model = model_from_json(model)
    model.load_weights('best_weights_256.hdf5')
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        mask = find_ball(img, model)
        img[mask > 220] = 255
        cv2.imshow("g",img)
        # cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)

if __name__ == '__main__':
    main()

"""This program was written for the purposes of converting gif files to png for the Carvana Image masking challenge:
https://www.kaggle.com/c/carvana-image-masking-challenge

By Jack Ellis
"""

import os
from PIL import Image

def iter_frames(im):
    try:
        i= 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0: 
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass

def convert(directory):
    """Converts gifs to PNG files
    
    Args:
        A directory
    """
    for filename in os.listdir(directory):
        if filename[-2:] != "py":
            im = Image.open(filename)
            for i, frame in enumerate(iter_frames(im)):
                frame.save("{}.jpg".format(filename[:-4]),**frame.info)

def deleteOld(directory):
    for i in range(len(os.listdir(directory))):
        if os.listdir(directory)[i][-3:] == "bmp":
            os.remove(os.listdir(directory)[i])


def main(directory):
    """Main Function

    Args:
        A directory.
    """
    convert(directory)
    deleteOld(directory)

if __name__ == '__main__':
    main(os.getcwd())

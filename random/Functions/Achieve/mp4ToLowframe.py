import cv2
from PIL import Image
from os import listdir
from os.path import isfile, join
import re
import shutil
import numpy as np


def import_lowframe(path_video, path_save_frames):
    vidcap = cv2.VideoCapture(path_video)
    success, image = vidcap.read()
    count = 0
    transposted_image = np.flip(np.moveaxis(image, 0,1),0)
    while success:
        success,image = vidcap.read()
        if count % 5 == 0:
            cv2.imwrite(path_save_frames + "frame%d.jpg" % count, transposted_image)     # save frame as JPEG file
            transposted_image = np.flip(np.moveaxis(image, 0,1),0)
            print('Read a new frame: ', success)
        count += 1

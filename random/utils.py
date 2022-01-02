import struct
import numpy as np
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
from keras.backend import expand_dims
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from os import listdir
from os.path import isfile, join
import re
import shutil


from yolo3_one_file_to_detect_them_all import *


def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


def draw_boxes(data, v_boxes, v_labels, v_scores):
    # convert to numpy
    
    # plot the image
    plt.figure(figsize = (12,8))
    plt.imshow(np.flip(data, -1))
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        plt.text(x1, y1+20, label, color='white')
    # show the plot
    plt.show()
    
    

def import_lowframe(path_video, path_save_frames, skip_step = 5):
    '''
    Takes the video file and creates low-frame pics. 
    '''
    vidcap = cv2.VideoCapture(path_video)
    success, image = vidcap.read()
    count = 0
    transposted_image = np.flip(np.moveaxis(image, 0,1),0)
    while success:
        success,image = vidcap.read()
        if count % skip_step == 0:
            cv2.imwrite(path_save_frames + "frame%d.jpg" % count, transposted_image)     # save frame as JPEG file
            transposted_image = np.flip(np.moveaxis(image, 0,1),0)
            print('Read a new frame: ', success)
        count += 1

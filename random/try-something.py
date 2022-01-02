# -*- coding: utf-8 -*-
"""
Spyder Editor

load video file, split into frames, crop images

"""

import cv2

path_video = "BikeVideo/FILE0003.MP4"

vidcap = cv2.VideoCapture(path_video)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
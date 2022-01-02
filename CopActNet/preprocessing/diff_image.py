#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 08:39:07 2021

@author: tenra
"""

from CopActNet.tools.reader import load_pickle
import CopActNet.config as cfg
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def diff_image_plot(video):

    # Only looking at the bottom of the picture (only the bicycle-path)
   # plt.imshow(video[600, 50:])

    # %% Trying to calculate the difference between frame only in the bottom part of the image
    nFrames = video.shape[0]
    lag, sigma  = 1, 10
    diffs = np.zeros(nFrames-lag)
    for i in range(lag, nFrames):
        diffs[i-lag] = np.sum(np.abs(video[i-lag,50:] - video[i, 50:]))

    # Applying gaussian 1d filter with 2 different sigmas
    fr, to = 0, 4800
    plt.figure(figsize = (12,8))
    plt.plot(diffs[fr:to], label = 'Absolute frame difference')
    filter1 = gaussian_filter1d(diffs, sigma)[fr:to]
    filter2 = gaussian_filter1d(diffs, 500)[fr:to]*1.05

    plt.plot(filter1, label = 'Low sigma')
    plt.plot(filter2, label = 'High sigma')
    plt.legend()
    plt.show()

def diff_image(video):

    # Only looking at the bottom of the picture (only the bicycle-path)
    # plt.imshow(video[600, 50:])

    # %% Trying to calculate the difference between frame only in the bottom part of the image
    nFrames = video.shape[0]
    lag, sigma  = 1, 10
    diffs = np.zeros(nFrames-lag)
    for i in range(lag, nFrames):
        diffs[i-lag] = np.sum(np.abs(video[i-lag,50:] - video[i, 50:]))
    
    fr, to = 0, video.shape[0]
    filter1 = gaussian_filter1d(diffs, sigma)[fr:to]
    filter2 = gaussian_filter1d(diffs, 500)[fr:to]*1.05
    
    diff = filter2 < filter1
    
    frames_return = np.hstack([np.array([False]),np.array(diff)])

    return video[frames_return]
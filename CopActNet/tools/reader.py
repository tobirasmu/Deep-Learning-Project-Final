# -*- coding: utf-8 -*-

from CopActNet.tools import reader
from CopActNet import config as cfg
from pathlib import Path, PosixPath
import cv2
import numpy as np
from tqdm import tqdm
import pickle


# %%
def load_video(path: PosixPath, every: int=1):
    """
    Loads a video using a given path. The frames are scaled down to the size given in the configuration and by the variable "every" in the frame dimension

    Parameters
    ----------
    path : PosixPath
        The path to the file.
    every : int
        Integer indicating how many frames should be "jumped" (every - 1)

    Returns
    -------
    frames : np.ndarray
        A tensor of shape Frames X Height X Width X Channels

    """
    cap = cv2.VideoCapture(str(path))
    F = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.empty((int(np.ceil(F / every)), cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH, 3), dtype=np.uint8)  # W and H changed because frame is rotated
    
    framesLoaded = 0
    for i in tqdm(range(F)):
        status, frame = cap.read()
        if status is False:
            break
        if (i % every) != 0:
            continue
        frame = np.flip(np.moveaxis(frame, 0, 1), 0)  # Rotation 90 degrees
        frame = cv2.resize(frame, dsize=(cfg.FRAME_WIDTH, cfg.ORIG_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        frame = frame[-cfg.FRAME_HEIGHT:, :, :]
        frames[framesLoaded, :, :, :] = np.array(frame)
        framesLoaded += 1
    cap.release()
    return frames


def load_pickle(path: PosixPath):
    """
    Loads an already pickled object

    Parameters
    ----------
    path : PosixPath
        The path to the pickled object

    Returns
    -------
    obj : TYPE
        The pickled object

    """
    file_handler = open(path, 'rb')
    obj = pickle.load(file_handler)
    file_handler.close()
    return obj

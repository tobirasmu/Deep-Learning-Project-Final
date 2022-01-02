#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:31:13 2021

@author: tenra
"""

from CopActNet.preprocessing import diff_image
from CopActNet.tools import reader, writer
from CopActNet import config as cfg
from tensorflow.keras.models import load_model
from CopActNet.models.yolo_model import predict_YOLO, predict_YOLO3_cphdata, predict_print_YOLO3_cphdata
from pathlib import Path
import numpy as np
from tqdm import tqdm

# %% Function for making the frames for annotation
def makeAnnotationFrames(video: np.ndarray):
    """
    Takes in a video as a numpy-tensor of size (Frames x Height x Width x Channels). Runs the pre-processing scheme.

    Parameters
    ----------
    video : np.ndarray
        Video given as a 4-dimensional numpy tensor.

    Returns
    -------
    np.ndarray
        The resulting video as a numpy-tensor including the subset of frames that are relevant to the analysis.

    """
    
    # Scaling down video using gaussian filtering
    active_frames = diff_image.diff_image(video)
    N_active = active_frames.shape[0]
    
    # Loading the YOLO model
    yolo_model = load_model(Path.joinpath(cfg.ROOT_DIR, "Data/model.h5"))
    
    print("Annotating the frames of the video using pre-trained YOLO-model")
    # Only looking at the frames that have either person or bicyclists
    annotation_indices = np.zeros(N_active, dtype="uint8")
    
    for i in tqdm(range(N_active)):
        img = active_frames[i]
        labels = predict_YOLO3_cphdata(img, yolo_model)
        if ("bicycle" in labels) or ("person" in labels):
            annotation_indices[i] = 1
    return active_frames[annotation_indices == 1]

# %%
if __name__ == "__main__":
    """
    This file only works if user have previously loaded a raw video using the functions in reader or just by running the "load_and_pickle_videos.py". After having 
    run this, the configuration variable "PICKLED_VIDEO_PAHTS" will be a list of the loaded and pickled video names.
    """
    # Path to pickled video
    path = cfg.PICKLED_VIDEO_PATHS[0]
    video = reader.load_pickle(path)
    
    # Only taking the subset we are interested in
    new_video = makeAnnotationFrames(video)
    print(f"Frames for annotation saved to the folder {path.stem} in the Data directory.")
    writer.tensor2frames_jpg(new_video, Path.joinpath(cfg.ROOT_DIR, "Annotations"), path.stem)
    writer.pickle_obj(Path.joinpath(cfg.ROOT_DIR, f"Annotations/{path.stem}_annotated.pkl"), new_video)
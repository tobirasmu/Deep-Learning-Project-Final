# -*- coding: utf-8 -*-
from pathlib import Path
import os


# %% Directories 

# Path to the "Deep-Learning-Project" folder
ROOT_DIR = Path(__file__).absolute().parent.parent

# Directories for the raw data
RAW_DATA_DIR = Path("/work1/fbohy/_Data for Frederik and Chris/Copenhagen Bike data")
RAW_VIDEO_PATHS = [Path.joinpath(RAW_DATA_DIR, file) for file in os.listdir(RAW_DATA_DIR) if file[-4:] == ".MP4"]

# Directories for pickled data
PICKLED_DIR = Path.joinpath(ROOT_DIR, "Data/pickled_videos")
PICKLED_VIDEO_PATHS = [Path.joinpath(PICKLED_DIR, file) for file in os.listdir(PICKLED_DIR)]

# %% Trained model paths
HELMNET_WEIGHTS_PATH = Path.joinpath(ROOT_DIR, "CopActNet/models/HelmNet_weights")
HELMNET_RESULTS_DIR = Path.joinpath(ROOT_DIR, "Results_helmnet")
HELMNET_RESULTS_PATH = Path.joinpath(HELMNET_RESULTS_DIR, "active_learning_results.txt")

# %% Video configurations
FRAME_RATE = 30.0
FRACTION_OF_IMAGE = 3/4
ORIG_FRAME_HEIGHT, FRAME_WIDTH = 171, 128
FRAME_HEIGHT = int(ORIG_FRAME_HEIGHT * FRACTION_OF_IMAGE)


# %% Dictionaries

HC_DICT = {(0, 0) : 0, 
           (1, 0) : 1,
           (1, 1) : 2,
           (2, 0) : 3,
           (2, 1) : 4, 
           (2, 2) : 5,
           (3, 0) : 6,
           (3, 1) : 7,
           (3, 2) : 8,
           (3, 3) : 9,
           (1, 2) : 10,
           (1, 3) : 11}
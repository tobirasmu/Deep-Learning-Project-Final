import cv2
import numpy as np
import torch as tc
import os
import CopActNet.config as cfg
from pathlib import Path, PosixPath
import pickle


def pickle_obj(path: PosixPath, obj):
    """
    Saves an object using pickle. 

    Parameters
    ----------
    path : PosixPath
        The desired location of the pickled object (including name) and with the .pkl extension
    obj : TYPE
        The object to be pickled

    Returns
    -------
    None.

    """
    file_handler = open(path, 'wb')
    pickle.dump(obj, file_handler)
    file_handler.close()
    


def writeTensor2video(x, name, out_directory=None):
    """
    Writes a tensor of shape (ch, num_frames, height, width) or (num_frames, height, width) (for B/W) to a video at the
    given out_directory. If no directory is given, it will just be placed in the current directory with the name.
    """
    if x.type() != 'torch.ByteTensor':
        x = x.type(dtype=tc.uint8)
    if out_directory is None:
        out_directory = os.getcwd() + '/'
    name = name + '.avi'
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        ch, num_frames, height, width = x.shape
    else:
        ch, num_frames, height, width = x.shape
    writer = cv2.VideoWriter(out_directory + name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cfg.FRAME_RATE,
                             (width, height))
    for i in range(num_frames):
        frame = np.moveaxis(x[:, i, :, :].type(dtype=tc.uint8).numpy(), 0, -1)
        if ch == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()


def tensor2frames_jpg(video: np.ndarray, path: PosixPath, name: str):
    """
    Makes a video (numpy array of size frames x height x width x channels) into a folder with all the frames as .jpg files. 
    
    Parameters
    ----------
    path : PosixPath
        The desired location of the folder
    video : np.ndarray
        The video that needs to be saved to a folder of frames
    name : str
        Name of the folder in which the frames should be saved.
    
    Returns
    -------
    None.
    
    """
    # Making a folder with the name of the video
    new_path = Path.joinpath(path, name)
    if os.path.isdir(new_path):
        raise FileExistsError("File already exists")
    os.mkdir(new_path)
    for i, frame in enumerate(video):
        cv2.imwrite(str(Path.joinpath(new_path, f"FRAME{i:0>5d}.jpg")), frame)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# -*- coding: utf-8 -*-

from CopActNet.tools import reader, writer
from pathlib import Path, PosixPath
from CopActNet import config as cfg


def load_and_pickle_video(path: PosixPath, every: int=1):
    """
    Loads and pickles video at the given location

    Parameters
    ----------
    path : PosixPath
        Location of raw video
    every : int, optional
        Integer used to specify how many frames should be "jumped" (every-1). The default is 1 (taking all frames - jumping none)

    Returns
    -------
    None.

    """
    # Loading video as a tensor
    video = reader.load_video(path, every)
    # Saving as a pickle obj with same name
    savepath = Path.joinpath(cfg.PICKLED_DIR, path.stem + ".pkl")
    writer.pickle_obj(savepath, video)
    print(f"Loaded the file {path.stem} and saved it to the location:\n{savepath}")
    

# %% 
if __name__ == "__main__":
    for i in range(3):
        load_and_pickle_video(cfg.RAW_VIDEO_PATHS[i], 3)
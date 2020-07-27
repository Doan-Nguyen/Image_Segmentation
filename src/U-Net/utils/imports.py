import os 
from os.path import splitext
import numpy as np 
from glob import glob
import logging 
from PIL import Image

import torch
from torch.utils.data import Dataset


class BasicDatasets(Dataset):
    """ This class to preprocess datasets to train. Take:
        - __len__(): the number of images (mask images)
        - __getitem__(): 
        - preprocess() ~ 
    """
    def __init__(self, imgs_dir, masks_dir, scale=1.0):
        """
        Parameters:
            - imgs_dir: the image's path
            - masks_dir: the masks image's path
            - scale: to resize images (0 < scale <= 1)
        """
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale

        assert 0 < scale <= 1, "Scale must be between 0 & 1"
        self.ids = [splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)
import os 
from os.path import splitext
import numpy as np 
from glob import glob
import logging 
from PIL import Image
import matplotlib.pyplot as plt

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

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size 
        
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        ### HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        ### take image & mask image by get detail paths
        for image in os.listdir(self.imgs_dir):
            mask_name = image.replace('.jpg', 'png')
            img_path = os.path.join(self.imgs_dir, image)
            mask_path = os.path.join(self.masks_dir, mask_name)

            mask = Image.open(mask_path)
            img = Image.open(img_path)

            assert img.size == mask.size, \
                logging.warning(f'Origin image {img_path} different size with mask images {mask_path}')
                # (f'Origin image {img_path} different size with mask images {mask_path}')

            img = self.preprocess(img, scale=1.0)
            mask = self.preprocess(mask, scale=1.0)

            return {
                'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor)
            }

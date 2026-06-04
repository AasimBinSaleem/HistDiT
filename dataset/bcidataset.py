# -*- coding: utf-8 -*-
# @Author : Xuanhe Er
# @Time   : 13/07/2023 10:57

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
from torchvision.io import read_image
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class BCIDataset(Dataset):
    def __init__(self, x_img_dir, y_img_dir, num_pairs=None, transforms=None):
        # data loading
        self.x_img_dir = x_img_dir
        self.x_dir_list = os.listdir(x_img_dir)
        self.x_dir_list = [x for x in self.x_dir_list if x.lower().endswith('.png')]
        self.x_dir_list.sort()

        self.y_img_dir = y_img_dir

        self.transforms = transforms

        self.num_pairs = num_pairs

    def get_number_files(self):
        return len([name for name in self.x_dir_list if os.path.isfile(os.path.join(self.x_img_dir, name))])

    def __getitem__(self, index):
        # read H&E
        x_sample_path = os.path.join(self.x_img_dir, self.x_dir_list[index])
        #print(x_sample_path)
        x = self.pil_loader(x_sample_path)
        # x = x.type(torch.FloatTensor) # 1shit code

        # read IHC
        y_sample_path = os.path.join(self.y_img_dir, self.x_dir_list[index])
        #print(y_sample_path)
        y = self.pil_loader(y_sample_path)
        
        # y = y.type(torch.FloatTensor) # 1shit code

        if self.transforms:
            # x = x.float().div(255) # 2shit code
            # y = y.float().div(255) # 2shit code
            x = self.transforms(x)
            y = self.transforms(y)
        return y, x

    def __len__(self):
        return self.get_number_files() if self.num_pairs is None else self.num_pairs

    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

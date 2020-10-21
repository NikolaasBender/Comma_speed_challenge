from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import hashlib
import csv


class ImageToSpeedDataset(Dataset):

    def __init__(self, csv, root_dir, transform=None):
        self.data = pd.read_csv(csv)
        print(len(self.data), " bits of data")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()

        speed, img1_name, img2_name = self.data.iloc[idx]
        img1_name = os.path.join(self.root_dir, img1_name)
        img2_name = os.path.join(self.root_dir, img2_name)
        img1 = io.imread(img1_name)
        img2 = io.imread(img2_name)
        sample = {'speed': speed, 'img1': img1, 'img2': img2}

        return sample

    def validate(self,
                 model,
                 txt_path="speedchallenge/data/train.txt",
                 vid_path="speedchallenge/data/train.mp4"):
        model.eval()
        vidcap = cv2.VideoCapture(vid_path)
        f = open(txt_path, "r")
        success, prev_img = vidcap.read()
        loss = 0.0
        while success:
            extra = extra_const
            success, image = vidcap.read()
            if success:
                speed = float(f.readline())
                out = model(prev_img, image)
                loss += loser(out.squeeze(), data["speed"].cuda().float())

        return loss

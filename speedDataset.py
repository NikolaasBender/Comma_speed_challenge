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
import random
import cv2


class ImageToSpeedDataset(Dataset):

    def __init__(self, csv, root_dir, transform=None):
        self.data = pd.read_csv(csv)
        print(len(self.data), " bits of data")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def smashData(self, prev_img, image):
        e = random.randint(1, 7)
        randomlist = []
        for i in range(0, e):
            n = random.randint(0,3)
            randomlist.append(n)
        for r in randomlist:
            if r == 0:
                # print("flip")
                prev_img = cv2.flip(prev_img, 1)
                image = cv2.flip(image, 1)
            elif r == 1:
                # print("random crop")
                prev_img, image = self.getRandomCrop(prev_img, image, 240, 240)
            elif r == 2:
                # print("noise")
                prev_img, image = self.colorJitter(prev_img, image)
            elif r == 3:
                prev_img, image = self.randomRotate(prev_img, image)
        return prev_img, image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()

        speed, img1_name, img2_name = self.data.iloc[idx]
        img1_name = os.path.join(self.root_dir, img1_name)
        img2_name = os.path.join(self.root_dir, img2_name)
        img1 = io.imread(img1_name)
        img2 = io.imread(img2_name)
        img1, img2 = self.smashData(img1, img2)
        sample = {'speed': speed, 'img1': img1, 'img2': img2}

        return sample

    def validate(self,
                 model,
                 loser,
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

    def getRandomCrop(self, image1, image2, crop_height, crop_width):

        max_x = image1.shape[1] - crop_width
        max_y = image1.shape[0] - crop_height

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        crop1 = image1[y: y + crop_height, x: x + crop_width]
        crop2 = image2[y: y + crop_height, x: x + crop_width]
        # print(crop.shape)

        return crop1, crop2


    def randomRotate(self, img1, img2):
        h, w, c = img1.shape
        r = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((w/2, h/2), r, 1)
        dst1 = cv2.warpAffine(img1, M, (w, h))
        dst2 = cv2.warpAffine(img2, M, (w, h))

        return dst1, dst2


    def colorJitter(self, img1, img2):
        h, w, c = img1.shape
        noise = np.random.randint(0, 50, (h, w))  # design jitter/noise here
        zitter = np.zeros_like(img1)
        zitter[:, :, 1] = noise
        noise_added1 = cv2.add(img1, zitter)
        noise_added2 = cv2.add(img2, zitter)
        return noise_added1, noise_added2

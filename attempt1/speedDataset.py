from __future__ import print_function, division
import os
import torch
import pandas as pd

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


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()

        _, speed, img1_name, img2_name = self.data.iloc[idx]
        img1_name = os.path.join(self.root_dir, img1_name)
        img2_name = os.path.join(self.root_dir, img2_name)
        img1 = cv2.imread(img1_name)
        img2 = cv2.imread(img2_name)
        
        img1, img2 = self.smashData([img1, img2])
        sample = {'speed': speed, 'img1': img1, 'img2': img2}

        return sample

    def validate(self,
                 m,
                 loser,
                 txt_path="speedchallenge/data/train.txt",
                 vid_path="speedchallenge/data/train.mp4"):
        # print("in val")
        vidcap = cv2.VideoCapture(vid_path)
        # print('opened video capture')
        f = open(txt_path, "r")
        # print('opened text')
        success, prev_img = vidcap.read()
        speed = float(f.readline())
        loss = 0.0
        # print('in validate')
        while success:
            success, image = vidcap.read()
            if success:
                    # print("validating")
                speed = float(f.readline())
                speed = torch.tensor(speed).cuda()
                # print(speed)
                out = m(torch.transpose(self.process(prev_img), 3, 1).cuda().float(), torch.transpose(self.process(image), 3, 1).cuda().float())
                # print(out)
                loss += loser(out.squeeze(), speed).item()

        return loss


    def smashData(self, q):
        e = random.randint(1, 7)
        randomlist = []
        for i in range(0, e):
            n = random.randint(0, 4)
            randomlist.append(n)
        
        for r in randomlist:
            if r == 0:
                # print("flip")
                q = self.flips(q)
            elif r == 1:
                # print("random crop")
                q = self.getRandomCrop(q, 0.90, 240)
            elif r == 2:
                # print("noise")
                q = self.colorJitter(q)
            elif r == 3:
                q = self.randomRotate(q)
            elif r == 4:
                pass
        q2 = []
        for d in q:
            q2.append(self.process(d))
        # print(randomlist, q2[0].shape, q2[1].shape)
        return q2


    def getRandomCrop(self, q, crop_percent, crop_min):

        if q[0].shape[1] <= crop_min + 5:
            return q
        

        max_x = max(q[0].shape[1] - q[0].shape[1] * crop_percent, crop_min + 5)
        max_y = max(q[0].shape[0] - q[0].shape[0] * crop_percent, crop_min + 5)

        crop = random.randint(0, int(q[0].shape[1] * crop_percent))

        q2 = []

        try:
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            for d in q:
                c = d[y: y + crop, x: x + crop]
                if c.shape[0] >= crop_min and c.shape[1] >= crop_min:
                    q2.append(c)
                else:
                    q2.append(d)
        except:
            print('error in random crop')
        

        return q2


    def randomRotate(self, q):
        h, w, c = q[0].shape
        r = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((w, h), r, 1)
        q2 = []
        for d in q:
            q2.append(cv2.warpAffine(d, M, (w, h)))
        
        return q2


    def colorJitter(self, q):
        h, w, c = q[0].shape
        noise = np.random.uniform(0, 1, 3)  # design jitter/noise here
        q2 = []
        for d in q:
            q2.append(d * noise)
        
        return q2

    def flips(self, q):
        q2 = []
        for d in q:
            q2.append(cv2.flip(d, 1))
        
        return q


    def process(self, img):
        img = cv2.resize(img, (240, 240))
        # img = img/255.0
        # img = img - np.array([0.485, 0.456, 0.406])
        # img = img/np.array([0.229, 0.224, 0.225])   # (shape: (256, 256, 3))
        
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        # if len(img.shape) < 4:
        #     img = img.unsqueeze(0)
        if list(img.shape) != [240, 240, 3]:
            print('theres an error with an image')
        img = torch.transpose(img, 0, 2)
        return img
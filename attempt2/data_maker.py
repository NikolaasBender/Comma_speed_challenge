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
import cv2
import random
import sys
import time
from multiprocessing import Process, Manager, Lock

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


csv_file = '../data/im_im_sp.csv'


def writeData(q, l):
    err = False
    names = []
    speeds = []
    for d in q:
        name = str(time.time() * random.random()) + ".jpg"
        # img = process(d[1])
        cv2.imwrite('../data/images/' + name, d[1])
        names.append(name)
        speeds.append(d[0])

    l.acquire()
    try:
        if err == False:
            with open(csv_file, '+a') as csvfile:
                writer = csv.writer(csvfile)
                if speeds[1] <= 3.0:
                    for i in range(4):
                        writer.writerow(speeds + names)
                writer.writerow(speeds + names)
    except:
        print("error writing the csv")
        exit()
    l.release()


def validate(img_folder="../data/images/",
             csv="../data/im_im_sp.csv"):
    data = pd.read_csv(csv)
    for _, speed, img1_name, img2_name in data.data.iterrows():
        img1_name = os.path.join(img_folder, img1_name)
        img2_name = os.path.join(img_folder, img2_name)
        img1 = cv2.imread(img1_name)
        img2 = cv2.imread(img2_name)
        try:
            img1 = cv2.resize(img1, (240, 240))
            img2 = cv2.resize(img2, (240, 240))
        except Exception as e:
            print(str(e))


def dataGoBrrrr(extra,
                txt_path="../speedchallenge/data/train.txt",
                vid_path="../speedchallenge/data/train.mp4",
                img_folder="../data/images/",
                csv="../data/im_im_sp.csv"):
    # initialize the que for storing data
    q = []
    vidcap = cv2.VideoCapture(vid_path)
    f = open(txt_path, "r")

    success, t_0 = vidcap.read()
    speed_0 = float(f.readline())

    q.append((speed_0, t_0))

    c = 0
    extra_const = extra
    lock = Lock()
    while success:
        if len(q) == 2:
            q.pop()
        extra = extra_const
        success, image = vidcap.read()
        if success:
            speed = float(f.readline())
            q.append((speed, image))

            writeData(q, lock)


start = time.time()
dataGoBrrrr(1)
print((time.time() - start)/60)

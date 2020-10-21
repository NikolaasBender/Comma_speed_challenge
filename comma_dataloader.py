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


csv_file = 'data/im_im_sp.csv'


def writeData(speed, img1, img2, l):
    err = False
    try:
        img1_name = str(time.time() * random.random()) + ".jpg"
        # print("hashed 1")
        img2_name = str(time.time() * random.random()) + ".jpg"
        # print("hashed 2")
    except:
        err = True
        print("error hashing images ", sys.exc_info())
        exit()

    try:
        img1 = process(img1)
        img2 = process(img2)
        cv2.imwrite('data/images/' + img1_name, img1)
        cv2.imwrite('data/images/' + img2_name, img2)
    except:
        err = True
        print("failed to write images")
        exit()
    l.acquire()
    try:
        if err == False:
            with open(csv_file, '+a') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow(['speed', 'image1', 'image2'])
                writer.writerow([speed, img1_name, img2_name])
    except:
        print("error writing the csv")
        exit()
    l.release()

def process(img):
    img = cv2.resize(img, (240, 240))
    # img = img/255.0
    # img = img - np.array([0.485, 0.456, 0.406])
    # img = img/np.array([0.229, 0.224, 0.225])   # (shape: (256, 256, 3))
    # img = img.astype(np.float32)
    return img


def getRandomCrop(image1, image2, crop_height, crop_width):

    max_x = image1.shape[1] - crop_width
    max_y = image1.shape[0] - crop_height

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    crop1 = image1[y: y + crop_height, x: x + crop_width]
    crop2 = image2[y: y + crop_height, x: x + crop_width]
    # print(crop.shape)

    return crop1, crop2


def randomRotate(img1, img2):
    h, w, c = img1.shape
    r = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((w/2, h/2), r, 1)
    dst1 = cv2.warpAffine(img1, M, (w, h))
    dst2 = cv2.warpAffine(img2, M, (w, h))

    return dst1, dst2


def colorJitter(img1, img2):
    h, w, c = img1.shape
    noise = np.random.randint(0, 50, (h, w))  # design jitter/noise here
    zitter = np.zeros_like(img1)
    zitter[:, :, 1] = noise
    noise_added1 = cv2.add(img1, zitter)
    noise_added2 = cv2.add(img2, zitter)
    return noise_added1, noise_added2


def smashData(speed, prev_img, image, l):
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
            prev_img, image = getRandomCrop(prev_img, image, 240, 240)
        elif r == 2:
            # print("noise")
            prev_img, image = colorJitter(prev_img, image)
        elif r == 3:
            prev_img, image = randomRotate(prev_img, image)
        # elif e == 4:
        #     img1, img2 = colorJitter(prev_img, image)
        #     img1, img2 = randomRotate(img1, img2)
        # elif e == 5:
        #     img1, img2 = getRandomCrop(prev_img, image, 240, 240)
        #     img1, img2 = randomRotate(img1, img2)
        # elif e == 6:
        #     img1, img2 = colorJitter(prev_img, image)
        #     img1, img2 = getRandomCrop(img1, img2, 240, 240)
        # elif e == 7:
        #     img1, img2 = colorJitter(prev_img, image)
        #     img1, img2 = getRandomCrop(img1, img2, 240, 240)
        #     img1, img2 = randomRotate(img1, img2)
        
    writeData(speed, prev_img, image, l)


def dataGoBrrrr(extra,
                txt_path="speedchallenge/data/train.txt",
                vid_path="speedchallenge/data/train.mp4",
                img_folder="data/images/",
                csv="data/im_im_sp.csv"):
    vidcap = cv2.VideoCapture(vid_path)
    f = open(txt_path, "r")
    success, prev_img = vidcap.read()
    speed = float(f.readline())
    # current_batch = []
    c = 0
    extra_const = extra
    lock = Lock()
    while success:
        extra = extra_const
        success, image = vidcap.read()
        if success:
            speed = float(f.readline())
            writeData(speed, prev_img, image, lock)

            # if speed <= 3.0:
            #     # extra_const = extra
            #     extra = extra * 5

            # procs = []
            # for i in range(extra):
            #     p = Process(target=smashData, args=(speed, prev_img, image, lock))
            #     p.start()
            #     procs.append(p)

            # for p in procs:
            #     p.join()
                

start = time.time()
dataGoBrrrr(30)
print((time.time() - start)/60)
#!/usr/bin/env python3

# Prepare the data
import numpy as np
import cv2
import torch
import random
import time
import torch.optim as optim
import torch.nn as nn
import resnet_builds
from torchvision import transforms, datasets
import speedDataset
import logger

BATCH_SIZE = 100


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
speed_dataset = speedDataset.ImageToSpeedDataset(csv='data/im_im_sp.csv',
                                                 root_dir='data/images/')
dataset_loader = torch.utils.data.DataLoader(speed_dataset,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=10)

loser = nn.MSELoss()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
logger.loggyboi(device)

m = resnet_builds.resnet152()
optimizer = optim.Adadelta(m.parameters(), lr=1.0)

if torch.cuda.device_count() > 1:
    logger.loggyboi("Let's use", torch.cuda.device_count(), "GPUs!")
    m = nn.DataParallel(m)
m.to(device)
m.train()

epochs = 500
logger.loggyboi("ready to train")

start = time.time()

for e in range(epochs):
    try:
        logger.loggyboi(e)
        count = 0
        running_loss = 0.0
        for data in dataset_loader:
            try:
                loss = None
                img1 = torch.transpose(data['img1'], 3, 1).cuda().float()
                img2 = torch.transpose(data['img2'], 3, 1).cuda().float()
                output = m(img1, img2)
                loss = loser(output, data["speed"].unsqueeze(1).cuda().float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())

                count += 1
                if count % 1000 == 0:
                    d = time.time() - start
                    fps = (1000 * BATCH_SIZE)/d
                    time_left = (
                        (len(speed_dataset) - (count * BATCH_SIZE))/fps)/60
                    logger.loggyboi(running_loss, "epoch loss\n",
                                    d/60, "min since last update\n",
                                    time_left, "min left\n",
                                    fps, "frames per second\n",
                                    100 * (count/(len(speed_dataset)//BATCH_SIZE)), "%")
                    logger.loggyboi(
                        '==============================================================')
                    start = time.time()
            except:
                logger.loggyboi('training error')

        logger.loggyboi("=====================saving===================")
        try:
            torch.save(m, "speed_net.pth")
        except err:
            logger.loggyboi("there was an error saving speed_net. exiting")
            exit()
    except:
        logger.loggyboi("massive error")

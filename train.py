#!/usr/bin/env python3
# 125518
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
import os

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
print(device)

m = resnet_builds.resnet152()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    m = nn.DataParallel(m)
m.to(device)
m.train()

optimizer = optim.Adadelta(m.parameters(), lr=1.2)

epochs = 200
print("ready to train")

start = time.time()

for e in range(0, epochs):
    print(e)
    count = 0
    running_loss = 0.0
    for data in dataset_loader:
        loss = None
        img1 = torch.transpose(data['img1'], 3, 1).cuda().float()
        img2 = torch.transpose(data['img2'], 3, 1).cuda().float()
        # print(data['img1'].shape)
        # output = m(data['img1'], data['img2'])
        output = m(img1, img2)
        # sanity check, may have to change for bugatti
        # if output.data >= 300 or output.data <= -75:
        #     print("weird error", i, e)
        #     # output.data = pre_sped.data
        #     exit()
        loss = loser(output.squeeze(), data["speed"].cuda().float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())

        count += 1
        if count % 100 == 0:
            d = time.time() - start
            fps = (100 * BATCH_SIZE)/d
            time_left = ((len(speed_dataset) - (count * BATCH_SIZE))/fps)/60
            print(running_loss, "epoch loss\n",
                  d/60, "min since last update\n",
                  time_left, "min left\n",
                  fps, "frames per second\n",
                  100 * (count/(len(speed_dataset)//BATCH_SIZE)), "%")
            print('==============================================================')
            start = time.time()

    print("=====================saving===================")
    torch.save(m, str(e) + "_sai_net.pth")
    try:
        os.remove("sai_full_data.pt")

        chpk = {'epoch_loss': running_loss,
                'epoch': e,
                'model': m,
                'optimizer': optimizer
                }

        torch.save(chpk, "sai_full_data.pt")
    except:
        print("error saving all of the information")

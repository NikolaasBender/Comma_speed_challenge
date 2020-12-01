#!/usr/bin/env python3

# Prepare the data
import numpy as np
import math
import cv2
import torch
import random
import time
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
import speedDataset
import pathlib
import deep_vo
import sys
import matplotlib.pyplot as plt

# PID 747257

def plot_loss(loss):
    img = 'records/losses.png'
    file_to_rem = pathlib.Path(img)
    file_to_rem.unlink()
    plt.plot(range(len(loss)), loss)
    plt.savefig(img)

BATCH_SIZE = 100
epoch_0 = 0
full_data = "records/full_data.pt"

txt_path = "speedchallenge/data/train.txt"
vid_path = "speedchallenge/data/train.mp4"

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
speed_dataset = speedDataset.ImageToSpeedDataset(csv='../data/im_im_sp.csv',
                                                 root_dir='../data/images/')
dataset_loader = torch.utils.data.DataLoader(speed_dataset,
                                             batch_size=BATCH_SIZE, shuffle=True,
                                             num_workers=4)

n = math.ceil(len(speed_dataset) / BATCH_SIZE)
loser = nn.MSELoss()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

m = deep_vo.DeepVO(imsize1=240, imsize2=240, batch=BATCH_SIZE)
m.to(device)
# optimizer = optim.AdamW(m.parameters(), lr=0.001)
optimizer = optim.SGD(m.parameters(), lr=0.001, momentum=0.9)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    m = nn.DataParallel(m)

# If you want to load from a save point
if len(sys.argv) > 1:
    savepoint = torch.load(full_data)
    epoch_0 = savepoint['epoch']
    m = savepoint['model']
    optimizer = savepoint['optimizer']
    print('loading from check point at epoch:', epoch_0)

m.train()

epochs = 200
print("ready to train")

loss_history = []
eval_history = []

start = time.time()
for e in range(epoch_0, epochs):
    print(e)
    count = 0
    running_loss = 0.0
    for i in range(n):
        batch = next(iter(dataset_loader))
        # print(batch)
        loss = None
        loss = m.step([batch['img1'].cuda().float(),
                       batch['img2'].cuda().float()],
                      batch['speed'].cuda().float(),
                      optimizer)

        # scheduler.step()

        running_loss += float(loss.item())

        count += 1
        if count % 10 == 0:
            d = time.time() - start
            fps = (10 * BATCH_SIZE)/d
            time_left = ((len(speed_dataset) - (count * BATCH_SIZE))/fps)/60
            print(running_loss, "epoch loss\n",
                  d/60, "min since last update\n",
                  time_left, "min left\n",
                  fps, "frames per second\n",
                  100 * (count/(len(speed_dataset)//BATCH_SIZE)), "%")
            print('==============================================================')
            start = time.time()

    print("=====================saving===================")
    loss_history.append(running_loss)
    try:
        plot_loss(loss_history)
        print('saving stuff')
        file_to_rem = pathlib.Path(full_data)
        file_to_rem.unlink()
        file_to_rem = pathlib.Path("models/dvo.pth")
        file_to_rem.unlink()
        torch.save(m, "models/dvo.pth")
        print('removed file')
        chpk = {'epoch_loss': running_loss,
                'epoch': e,
                'model': m,
                'optimizer': optimizer
                }

        torch.save(chpk, full_data)
        print('saved checkpoint')
    except:
        print("error saving all of the information")

    print('loss history:', loss_history)

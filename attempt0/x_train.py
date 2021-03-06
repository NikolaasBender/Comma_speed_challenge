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
import pathlib
import xception

# PID 1987267

BATCH_SIZE = 30

txt_path="speedchallenge/data/train.txt"
vid_path="speedchallenge/data/train.mp4"


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

m = xception.xception(pretrained=False, device=device)
optimizer = optim.AdamW(m.parameters(), lr=0.001)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    m = nn.DataParallel(m)
m.to(device)
m.train()

epochs = 200
print("ready to train")

loss_history = []
eval_history = []

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
        
        output = m(img1, img2)
        
        # loss = loser(output.squeeze(), data["speed"].cuda().float())
        l = (output.squeeze() - data["speed"].cuda().float())**4
        loss = sum(l) + max(l)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    torch.save(m, str(e) + "_x_net.pth")
    loss_history.append(running_loss)
    try:
        print('saving stuff')
        file_to_rem = pathlib.Path("full_data.pt")
        file_to_rem.unlink()
        print('removed file')
        chpk = {'epoch_loss': running_loss,
                'epoch': e,
                'model': m,
                'optimizer': optimizer
                }

        torch.save(chpk, "full_data.pt")
        print('saved checkpoint')
    except:
        print("error saving all of the information")

    try:
        if e%10 == 0:
            error_in_val = speed_dataset.validate(m, loser)
            eval_history.append(error_in_val)
            print('validation:', error_in_val)
    except:
        print('error in validation')
        eval_history.append(None)
    
    print('loss history:', loss_history)
    print('eval history:', eval_history)

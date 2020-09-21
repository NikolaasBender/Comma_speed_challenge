#!/usr/bin/env python3

# Prepare the data
import numpy as np
import cv2
import torch
import prepper
import random
import resnet_builds
import time
import torch.optim as optim
import torch.nn as nn


loser = nn.MSELoss()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 4
threshold = 8
extra = 4

DM = prepper.DataManager(threshold, extra, batch_size)


m = resnet_builds.resnet152().to(device)
m.train()
optimizer = optim.Adadelta(m.parameters(), lr=1.0)

epochs = 200

start = time.time()

for e in range(0, epochs):
    DM.randomize()
    count = 0
    running_loss = 0.0
    for i in range(len(DM.data)//batch_size):
        batch = DM.batchGet()
        loss = None
        output = m(batch.x.to(device))
         # sanity check, may have to change for bugatti
        # if output.data >= 300 or output.data <= -75:
        #     print("weird error", i, e)
        #     # output.data = pre_sped.data
        #     exit()
        loss = loser(output, batch.y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())

        count += 1
        if count%100 == 0:
            d = time.time() - start
            fps = 100/d
            time_left = ((20400-count)/fps)/60
            print(running_loss, "epoch loss\n", 
            time_left, "min left\n", 
            fps, "frames per second",
            100 * (count/20400), "%")
            print('==============================================================')
            start = time.time()

    print("=====================saving===================")
    torch.save(m, str(e) + "_sai_net.pth")
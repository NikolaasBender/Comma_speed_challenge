#!/usr/bin/env python3
import cv2
from convyboi import Net
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from xception import xception

loser = nn.MSELoss()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
m = xception(device=device).to(device)
m.train()
optimizer = optim.Adadelta(m.parameters(), lr=0.1)

pre_sped = torch.tensor([[0.0]]).to(device)

epochs = 200

for e in range(epochs):
    vidcap = cv2.VideoCapture('/home/nick/projects/comma/speedchallenge/data/train.mp4')
    f = open("/home/nick/projects/comma/speedchallenge/data/train.txt", "r")
    success, prev_img = vidcap.read()
    count = 0
    running_loss = 0.0
    for i in range(1, 20400):
        success, image = vidcap.read()
        if success:
            loss = None
            diff = image - prev_img
            diff = cv2.resize(diff, (400, 400))
            # diff = cv2.resize(diff, (299, 299))
            diff = diff/255.0
            diff = diff - np.array([0.485, 0.456, 0.406])
            diff = diff/np.array([0.229, 0.224, 0.225])   # (shape: (256, 256, 3))
            diff = diff.astype(np.float32)
            diff = torch.from_numpy(diff).unsqueeze(0)
            diff = np.transpose(diff, (0, 3, 1, 2)).to(device)
            speed = float(f.readline())
            speed = torch.tensor([speed]).to(device)

            # Do an infrence to get speed
            output = m(diff, pre_sped)

            # sanity check, may have to change for bugatti
            if output.data >= 300 or output.data <= -75:
                print("weird error", i, e)
                # output.data = pre_sped.data
                exit()
            

            # print(output.item(), speed.item())
            loss = loser(output, speed)
            # print(loss)
            #  loss = abs(output[0] - speed)
            # print(loss)

            optimizer.zero_grad()
            if i == 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            # loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            count += 1
            prev_img = image
            pre_sped.data = output.data

            if count%100 == 0:
                print(running_loss)
                print(100 * (count/20400), "%")
        else:
            print("getting out of here")

    torch.save(m, str(e) + "two.pth")
    f.close()



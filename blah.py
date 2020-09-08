#!/usr/bin/env python3
import cv2
from convyboi import Net
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from xception import xception
import time

def process(img):
    img = cv2.resize(img, (480, 480))
    img = img/255.0
    img = img - np.array([0.485, 0.456, 0.406])
    img = img/np.array([0.229, 0.224, 0.225])   # (shape: (256, 256, 3))
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    return img


loser = nn.MSELoss()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
m = xception(device=device).to(device)
m.train()
optimizer = optim.Adadelta(m.parameters(), lr=0.1)

epochs = 200

start = time.time()

for e in range(epochs):
    vidcap = cv2.VideoCapture('/home/nick/projects/comma/speedchallenge/data/train.mp4')
    f = open("/home/nick/projects/comma/speedchallenge/data/train.txt", "r")
    success, prev_img = vidcap.read()
    speed = float(f.readline())
    count = 0
    running_loss = 0.0
    while success:
        success, image = vidcap.read()
        if success:
            loss = None
            
            # torch.tensor([[0.0]]).to(device)
            dat = torch.cat((process(image), process(prev_img)), dim=2).unsqueeze(0)
            # print(dat.size())
            dat = np.transpose(dat, (0, 3, 1, 2)).to(device)
            speed = float(f.readline())
            speed = torch.tensor([speed]).to(device)

            # Do an infrence to get speed
            output = m(dat)

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
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            count += 1
            prev_img = image

            

            if count%100 == 0:
                d = time.time() - start
                fps = 100/d
                time_left = ((20400-count)/fps)/60
                print(running_loss, "epoch loss\n", time_left, "min left\n", fps, "frames per second")
                print(100 * (count/20400), "%")
                print('==============================================================')
                start = time.time()
        else:
            print("getting out of here")

    print("=====================saving===================")
    torch.save(m, str(e) + "_xnet_6d.pth")
    f.close()
    vidcap.release()



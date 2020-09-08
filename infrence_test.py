#!/usr/bin/env python3

import cv2
from convyboi import Net
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from xception import xception


def process(img):
    img = cv2.resize(img, (480, 480))
    img = img/255.0
    img = img - np.array([0.485, 0.456, 0.406])
    img = img/np.array([0.229, 0.224, 0.225])   # (shape: (256, 256, 3))
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    return img

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
  
# org 
org = (50, 50) 
  
# fontScale 
fontScale = 1
   
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
m = xception(device=device, pretrained=True, path='/home/nick/projects/comma/9_xnet_6d.pth').to(device)
m.eval()

pre_sped = torch.tensor([[10.0]]).to(device)

vidcap = cv2.VideoCapture('/home/nick/projects/comma/speedchallenge/data/test.mp4')    
success, prev_img = vidcap.read()
while success:
    success, image = vidcap.read()
    dat = torch.cat((process(image), process(prev_img)), dim=2).unsqueeze(0)
    dat = np.transpose(dat, (0, 3, 1, 2)).to(device)

    # Do an infrence to get speed
    output = m(dat)

    # sanity check, may have to change for bugatti
    if output.data >= 300 or output.data <= -75:
        print("weird error", i, e)
        # output.data = pre_sped.data
        exit()

    prev_img = image
    pre_sped.data = output.data

    image = cv2.putText(image, str(output.item()), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    cv2.imshow('image', np.array(image, dtype = np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
else:
    print("getting out of here")

print("end")

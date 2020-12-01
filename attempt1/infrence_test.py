#!/usr/bin/env python3

import cv2
from convyboi import Net
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from xception import xception
import resnet_builds



def process(img):
    img = cv2.resize(img, (240, 240))
    img = img/255.0
    img = img - np.array([0.485, 0.456, 0.406])
    img = img/np.array([0.229, 0.224, 0.225])   # (shape: (256, 256, 3))
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
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
m = xception(pretrained=True, device=device, path='127_x_net.pth')
m.to(device)
m.eval()

vidcap = cv2.VideoCapture('/home/nick/projects/comma/speedchallenge/data/test.mp4')    
success, prev_img = vidcap.read()
while success:
    success, image = vidcap.read()
    img1 = process(image)
    img2 = process(prev_img)
    img1 = torch.transpose(img1, 3, 1).cuda().float()
    img2 = torch.transpose(img2, 3, 1).cuda().float()
    # print(data['img1'].shape)
    # output = m(data['img1'], data['img2'])
    output = m(img1, img2)

    # sanity check, may have to change for bugatti
    if output.data >= 300 or output.data <= -75:
        print("weird error", i, e)
        # output.data = pre_sped.data
        exit()

    prev_img = image

    image = cv2.putText(image, str(round(output.item(), 2)), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    cv2.imshow('image', np.array(image, dtype = np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
else:
    print("getting out of here")

print("end")

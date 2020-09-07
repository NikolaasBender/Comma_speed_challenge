# Prepare the data

import numpy as np
import cv2
import torch
import pickle

class Frame:
    def __init__(self, f0, f1, v, d):
        self.speed = torch.tensor([v]).to(d)
        self.device = d
        self.diff = self.process(f0, f1)
    
    def process(self, frame0, frame1):
        diff = frame0 - frame1
        diff = cv2.resize(diff, (400, 400))
        # diff = cv2.resize(diff, (299, 299))
        diff = diff/255.0
        diff = diff - np.array([0.485, 0.456, 0.406])
        diff = diff/np.array([0.229, 0.224, 0.225])   # (shape: (256, 256, 3))
        diff = diff.astype(np.float32)
        diff = torch.from_numpy(diff).unsqueeze(0)
        diff = np.transpose(diff, (0, 3, 1, 2)).to(self.device)
        return diff


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

DATA = []

vidcap = cv2.VideoCapture('/home/nick/projects/comma/speedchallenge/data/train.mp4')
t = open("/home/nick/projects/comma/speedchallenge/data/train.txt", "r")
success, prev_img = vidcap.read()
for i in range(1, 20400):
    success, image = vidcap.read()
    speed = float(t.readline())
    if success:
        f = Frame(prev_img, image, speed, device)
        DATA.append(f)
        prev_img = image

        if i%10 == 0:
            print(100 * (i/20400), "%")
    else:
        print("getting out of here")


# open a file, where you ant to store the data
file = open('important', 'wb')

# dump information to that file
pickle.dump(DATA, file)

# close the file
file.close()

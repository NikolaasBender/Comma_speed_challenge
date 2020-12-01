import torch
import numpy as np
import random
import cv2
import torchvision


def process(img):
    img = cv2.resize(img, (240, 240))
    img = img/255.0
    img = img - np.array([0.485, 0.456, 0.406])
    img = img/np.array([0.229, 0.224, 0.225])   # (shape: (256, 256, 3))
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    return img


def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


def colorJitter(img1, img2):
    h,w,c = img1.shape
    noise = np.random.randint(-0.5, 50, (h, w)) # design jitter/noise here
    zitter = np.zeros_like(img1)
    zitter[:,:,1] = noise
    noise_added1 = cv2.add(img1, zitter)
    noise_added2 = cv2.add(img2, zitter)
    return noise_added1, noise_added2


class batch:
    def __init__(self, data):
        images = [x.data for x in data]
        speeds = [x.speed for x in data]
        x = torch.stack(images)
        self.x = np.transpose(x, (0, 3, 1, 2))
        self.y = torch.stack(speeds)

class data:
    def __init__(self, im0, im1, v):         
        self.speed = torch.tensor([v])
        # Create a 6d image
        self.data = torch.cat((process(im1), process(im0)), dim=2)


def load(thresh, extra, txt_path="/home/nick/projects/comma/speedchallenge/data/train.txt", vid_path="/home/nick/projects/comma/speedchallenge/data/train.mp4"):
    vidcap = cv2.VideoCapture(vid_path)
    f = open(txt_path, "r")
    success, prev_img = vidcap.read()
    all_batches = []
    # current_batch = []
    c = 0
    while success:
        success, image = vidcap.read()
        if success:
            speed = float(f.readline())
            d = data(prev_img, image, speed)
            all_batches.append(d)

            t = random.randint(0,10)
            if t > thresh:
                for i in range(extra):
                    e = random.randint(0,2)
                    if e == 0:
                        d = data( cv2.flip(prev_img, 1), 
                        cv2.flip(image, 1), 
                        speed)
                    elif e == 1:
                        d = data( get_random_crop(prev_img, 240, 240), 
                        get_random_crop(image, 240, 240), 
                        speed)
                    elif e == 2:
                        i1, i2 = colorJitter(prev_img, image)
                        d = data(i1, i2, speed)
                    all_batches.append(d)
                    
            # current_batch.append(d)
            # if len(current_batch) == batch_len:
            #     b = batch(current_batch)
            #     all_batches.append(b)
            #     current_batch.clear()
            #     c += 1
            #     percent = 100 * (c*batch_len / 20400)
            #     if percent % 5 == 0:
            #         print(percent)
            #     # print(percent, '%',  "loaded into memory")
            #     # Limited ram testing
            #     # if percent >= 20.0:
            #     #     break
            

    print("returning batches")
    return all_batches


class DataManager:
    def __init__(self, threshold, extra, batch_size, **kwargs):
        self.data = load(threshold, extra)
        self.batch = batch_size
        self.n = 0

    def batchGetRand(self):
        random.shuffle(self.data)
        d = []
        for i in range(self.batch):
            d.append(self.data[i])
        return batch(d)
    
    def batchGet(self):
        d = []
        for i in range(self.batch):
            if i + self.n > len(self.data) - 1:
                self.n = 0
                break
            d.append(self.data[i+self.n])
        self.n += self.batch
        return batch(d)

    def randomize(self):
        random.shuffle(self.data)
        


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 478, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(478, 160, kernel_size=3, stride=3)
        self.conv3 = nn.Conv2d(160, 53, kernel_size=5, stride=3)
        self.conv4 = nn.Conv2d(53, 50, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(50, 25, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(25, 10, kernel_size=5, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc0 = nn.Linear(61, 50)
        self.fc1 = nn.Linear(50, 40)
        self.fc2 = nn.Linear(40, 1)
        self.speed = torch.tensor([0.0]).to(device)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = torch.flatten(x)
        # print(x.size())
        x = torch.cat((x, self.speed))
        x = self.fc0(x)
        x = F.relu(x)
        # ADD IN PREVIOUS SPEED
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        output = self.fc2(x)
        self.speed = output.detach()
        return output


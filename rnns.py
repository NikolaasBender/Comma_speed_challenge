import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np

rnn = nn.LSTM(input_size=10, hidden_size=20, num_layers=5)
input = torch.randn(10, 3, 10)
h0 = torch.randn(5, 3, 20)
c0 = torch.randn(5, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
print(output)
print(hn)
print(cn)
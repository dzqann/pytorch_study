#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/17 23:15 
# @Author : DZQ
# @File : 1.py

import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader

from torchvision import transforms as tfs
from torchvision.datasets import MNIST

data_tf = tfs.Compose(
    [
        tfs.ToTensor(),
        tfs.Normalize([0.5], [0.5])
    ]
)

train_set = MNIST("/day03/data", train=True, transform=data_tf)
test_set = MNIST("/day03/data", train=False, transform=data_tf)

train_data = DataLoader(train_set, 64, True, num_workers=1)
test_data = DataLoader(test_set, 128, False, num_workers=1)


class RNN(nn.Module):
    def __init__(self, in_feature=28, hidden_feature=100, num_class=10, num_layers=1):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)
        self.classifier = nn.Liner(hidden_feature, num_class)

    def forward(self, x):
        x = x.squeeze()
        x = x.permute(2, 0, 1)
        out, _ = self.rnn(x)
        out = out[-1, :, :]
        out = self.classifier(out)
        return out

net = RNN()
criterion = nn.CrossEntropyLoss()

optimzier = torch.optim.Adadelta(net.parameters(), 1e-1)
from utils import train
train(net, train_data, test_data, 10, optimzier, criterion)
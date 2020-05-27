#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/12 21:48 
# @Author : DZQ
# @File : MyLinerRegression.py

from torch.autograd import Variable
from torch import nn
from torch import optim
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import torch

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


class LineRegression(nn.Module):
    def __init__(self):
        super(LineRegression, self).__init__()
        self.lr = nn.Linear(1, 1)

    def forward(self, x):
        return self.lr(x)


x_data = Variable(torch.from_numpy(x_train), requires_grad=True)
y_data = Variable(torch.from_numpy(y_train), requires_grad=True)

model = LineRegression()
critisement = nn.MSELoss()
optimster = optim.SGD(model.parameters(), lr=1e-3)
steps = 1000
for each in range(steps):
    out = model(x_data)
    loss = critisement(out, y_data)
    optimster.zero_grad()
    loss.backward()
    optimster.step()
    if (each + 1) % 50 == 0:
        print("step: {}, loss: {:.4f}".format(each + 1, loss.data))

model.eval()
w1 = model.lr.weight.detach().numpy()[0][0]
b = model.lr.bias.detach().numpy()[0]

plt.plot(x_train, y_train, 'ro', label="origin data")

line = np.linspace(x_train.min(), x_train.max(), 200)
plt.plot(line, w1 * line + b, 'b', label="regression")
plt.legend()
plt.show()

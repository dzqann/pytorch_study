#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2020/5/13 21:55 
# @Author : DZQ
# @File : MNIST.py

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,))  # 拉平
    x = torch.from_numpy(x)
    return x


train_dataset = datasets.MNIST("./data", train=True, download=True, transform=data_tf)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=data_tf)

train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data = DataLoader(train_dataset, batch_size=128, shuffle=False)

net = nn.Sequential(
    nn.Linear(28 * 28, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-1)
losses = []
acces = []
eval_losses = []
eval_acces = []
for i in range(20):
    train_loss = 0
    train_acc = 0
    net.train()
    for image, label in train_data:
        image = Variable(image)
        label = Variable(label)
        out = net(image)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / image.shape[0]
        train_acc += acc
    losses.append(train_loss / len(train_dataset))
    acces.append(train_acc / len(train_data))
    eval_loss = 0
    eval_acc = 0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(i, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))

plt.figure(figsize=(20, 8), dpi=80)
plt.title("loss")
plt.plot(np.arange(len(losses)), losses)
plt.show()
plt.plot(np.arange(len(acces)), acces)
plt.title('train acc')
plt.show()
